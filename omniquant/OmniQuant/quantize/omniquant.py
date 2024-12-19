import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_dnabert_layer import QuantBertLayer
from torch.cuda.amp import autocast, GradScaler
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from .bert_padding import (index_first_axis,
                                            index_put_first_axis, pad_input,
                                            unpad_input, unpad_input_only)
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    print('------')
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    is_bert = False
    if "bert" in args.net.lower():
        is_bert = True
        layers = model.bert.encoder.layer
        model.bert.embeddings = model.bert.embeddings.to(dev)
        model.bert.pooler = model.bert.pooler.to(dev)
        EncoderLayer = QuantBertLayer
        pairs = {
            "Wqkv":"qkv",
            "wo":"out",
            "dense":"fc1"
        }
        layer_name_prefix = "bert.encoder.layer"
    elif "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
   
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_bert = False

        def forward(self, inp, cu_seqlens=None, seqlen=None,
                                             subset_idx=None,
                                             indices=None,
                                             attn_mask=None,
                                             bias=None,  **kwargs):
                                                 
            if is_bert:
                padded_tensor = torch.zeros(model.config.max_position_embeddings, 768)
                padded_mask = padded_mask = torch.zeros(model.config.max_position_embeddings, 1)
                # Copy the original tensor into the padded tensor
                padded_tensor[:inp.size(0)] = inp
                inps[cache["i"]] = padded_tensor
                attn_mask = attn_mask.unsqueeze(0) if attn_mask.dim() == 1 else attn_mask
                cache["i"] += 1
                if attn_mask != None:
                    padded_mask[:attn_mask.size(1)] = attn_mask.transpose(0, 1)
                cache["attention_mask"] = padded_mask if attn_mask != None else kwargs["attention_mask"]
            else:
                cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError    

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_bert = is_bert

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                if isinstance(batch, dict):
                    inputs = batch["input_ids"][0].unsqueeze(0).to(dev)
                    attention_mask=batch['attention_mask'][0].unsqueeze(0).to(dev)
                    model = model.to(dev)
                    with autocast():
                        model(input_ids=inputs, attention_mask=attention_mask)
                else:
                    inputs = batch[0].to(dev)
                    model(inputs)

            except ValueError:
                pass
    # move embedding layer and first layer to cpu
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "bert" in args.net.lower():
        model.bert.embeddings = model.bert.embeddings.cpu()
        model.bert.pooler.dense = model.bert.pooler.dense.cpu()
    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache['attention_mask']
    
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    attention_mask_batch = attention_mask_batch.to(dev)
    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        elif "bert" in args.net.lower():  
          qlayer = EncoderLayer(layer, args)  
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        if is_bert:
                            self = model.bert.encoder
                            
                            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
                            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                            attention_mask_bool = attention_mask.bool()
                            batch, seqlen = attention_mask.unsqueeze(0).shape[:2]
                            hidden_states, indices, cu_seqlens, _ = unpad_input(
                                fp_inps[j].unsqueeze(0), attention_mask_bool.to(dev))  
                            if self._current_alibi_size < seqlen:
                                    # Rebuild the alibi tensor when needed
                                warnings.warn(
                                    f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
                                )
                                self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
                            elif self.alibi.device != hidden_states.device:
                                # Device catch-up
                                self.alibi = self.alibi.to(hidden_states.device)
                            alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
                            attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
                            
                            alibi_attn_mask = attn_bias.to(dev) + alibi_bias
                            
                            fp_inps[j] = qlayer(hidden_states.to(dev),cu_seqlens.to(dev), seqlen, indices=indices.to(dev), attn_mask=attention_mask.to(dev),bias=alibi_attn_mask)[0]
                            if args.aug_loss:
                                batch, seqlen = quant_inps[j].unsqueeze(0).shape[:2]
                                hidden_states, indices, cu_seqlens, _ = unpad_input(
                                    quant_inps[j].unsqueeze(0), attention_mask_bool)
                                if self._current_alibi_size < seqlen:
                                    # Rebuild the alibi tensor when needed
                                    warnings.warn(
                                        f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
                                    )
                                    self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
                                elif self.alibi.device != hidden_states.device:
                                    # Device catch-up
                                    self.alibi = self.alibi.to(hidden_states.device)
                                alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
                                attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
                                alibi_attn_mask = attn_bias + alibi_bias
                                fp_inps_2[j] = qlayer(hidden_states, cu_seqlens, seqlen, indices=indices, attn_mask=attention_mask)[0]
                        else:
                            fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                            if args.aug_loss:
                                fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.attention.self.Wqkv.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama, is_bert)
                        if is_bert:
                            self = model.bert.encoder
                            
                            attention_mask_batch = attention_mask_batch.squeeze()
                            extended_attention_mask = attention_mask_batch.unsqueeze(1).unsqueeze(2)
                            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
                            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                            attention_mask_bool = attention_mask_batch.bool()
                            
                            batch, seqlen = attention_mask_batch.shape[:2]
                            hidden_states, indices, cu_seqlens, _ = unpad_input(
                                quant_inps[index:index+args.batch_size,], attention_mask_bool.to(dev))  

                            if self._current_alibi_size < seqlen:
                                    # Rebuild the alibi tensor when needed
                                warnings.warn(
                                    f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
                                )
                                self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
                            elif self.alibi.device != hidden_states.device:
                                # Device catch-up
                                self.alibi = self.alibi.to(hidden_states.device)
                            alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
                            attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
                            
                            alibi_attn_mask = attn_bias.to(dev) + alibi_bias
                           
                            quant_out = qlayer(hidden_states.to(dev),cu_seqlens.to(dev), seqlen, indices=indices.to(dev), attn_mask=attention_mask_batch.to(dev),bias=alibi_attn_mask)[0]

                        else:
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama, is_bert)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        if is_bert:
                            self = model.bert.encoder
                            attention_mask2 = attention_mask.view(1, -1)
                            extended_attention_mask = attention_mask2.unsqueeze(1).unsqueeze(2)
                            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
                            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                            attention_mask_bool = attention_mask2.bool()
                            
                            batch, seqlen = attention_mask2.shape[:2]
                            

                            hidden_states, indices, cu_seqlens, _ = unpad_input(
                                quant_inps[j].unsqueeze(0), attention_mask_bool.to(dev))  

                            
                            if self._current_alibi_size < seqlen:
                                    # Rebuild the alibi tensor when needed
                                warnings.warn(
                                    f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
                                )
                                self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
                            elif self.alibi.device != hidden_states.device:
                                # Device catch-up
                                self.alibi = self.alibi.to(hidden_states.device)
                            alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
                            attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
                            
                            alibi_attn_mask = attn_bias.to(dev) + alibi_bias

                           
                            quant_inps[j] = qlayer(hidden_states.to(dev).to(torch.float16),cu_seqlens.to(dev), 512, indices=indices.to(dev), attn_mask=attention_mask2.to(dev),bias=alibi_attn_mask)[0]
                        else:
                            quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4,6,8] and args.abits >= 4   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits > 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

