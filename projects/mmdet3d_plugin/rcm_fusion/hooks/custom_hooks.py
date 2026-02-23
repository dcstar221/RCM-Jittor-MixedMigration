from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time
from tqdm import tqdm

@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

@HOOKS.register_module()
class TqdmProgressHook(Hook):
    def __init__(self):
        self.pbar = None

    def before_epoch(self, runner):
        if runner.rank == 0:
            self.pbar = tqdm(total=len(runner.data_loader), desc=f"Epoch {runner.epoch+1}/{runner.max_epochs}")

    def after_train_iter(self, runner):
        if runner.rank == 0 and self.pbar is not None:
            self.pbar.update(1)
            # Optionally show loss in postfix
            if 'loss' in runner.outputs:
                loss_val = runner.outputs['loss']
                if hasattr(loss_val, 'item'):
                    loss_val = loss_val.item()
                self.pbar.set_postfix(loss=f"{loss_val:.4f}")

    def after_epoch(self, runner):
        if runner.rank == 0 and self.pbar is not None:
            self.pbar.close()
