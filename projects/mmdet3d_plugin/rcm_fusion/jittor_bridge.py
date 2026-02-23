import jittor as jt
import torch
import torch.utils.dlpack

def _jt_from_dlpack(dlpack_tensor, fallback_tensor=None):
    if hasattr(jt, "from_dlpack"):
        return jt.from_dlpack(dlpack_tensor)
    if hasattr(jt, "misc") and hasattr(jt.misc, "from_dlpack"):
        return jt.misc.from_dlpack(dlpack_tensor)
    if fallback_tensor is not None:
        return jt.array(fallback_tensor.detach().cpu().numpy())
    raise AttributeError("Jittor DLPack import is unavailable")

def torch2jittor(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, list):
        return [torch2jittor(t) for t in tensor]
    if isinstance(tensor, dict):
        return {k: torch2jittor(v) for k, v in tensor.items()}
    if isinstance(tensor, tuple):
        return tuple(torch2jittor(t) for t in tensor)
    if not isinstance(tensor, torch.Tensor):
        return tensor

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return _jt_from_dlpack(torch.utils.dlpack.to_dlpack(tensor), tensor)

def jittor2torch(array):
    if array is None:
        return None
    if isinstance(array, list):
        return [jittor2torch(a) for a in array]
    if isinstance(array, dict):
        return {k: jittor2torch(v) for k, v in array.items()}
    if isinstance(array, tuple):
        return tuple(jittor2torch(a) for a in array)
    if not isinstance(array, jt.Var):
        return array
    if hasattr(array, "dlpack"):
        return torch.utils.dlpack.from_dlpack(array.dlpack())
    return torch.from_numpy(array.numpy())

def sync_weights_pt_to_jt(pt_state, jt_module):
    """
    Copy all weights from a PyTorch module state dict to a Jittor module with matching names.
    This dynamically traverses the modules.
    """
    jt_state = jt_module.state_dict()
    synced = 0
    missing = []
    failed = []
    for name, pt_tensor in pt_state.items():
        # num_batches_tracked is a BatchNorm counter not used in Jittor
        if name.endswith('num_batches_tracked'):
            continue
        if name in jt_state:
            if not pt_tensor.is_contiguous():
                pt_tensor = pt_tensor.contiguous()
            try:
                jt_tensor = _jt_from_dlpack(torch.utils.dlpack.to_dlpack(pt_tensor.detach()), pt_tensor)
                jt_state[name].assign(jt_tensor)
                synced += 1
            except Exception as e:
                failed.append(f"  {name}: {str(e)}")
        else:
            missing.append(name)
    total = len(pt_state)
    print(f"   Weight sync: {synced}/{total} parameters synced successfully.")
    if missing:
        print(f"   Warning: {len(missing)} parameters missing in Jittor model:")
        for m in missing:
            print(f"     - {m}")
    if failed:
        print(f"   Warning: {len(failed)} parameters failed to sync:")
        for f_msg in failed:
            print(f_msg)

