# """
# Modified from https://github.com/KaiyangZhou/deep-person-reid
# """
# import torch
# from torch.optim.lr_scheduler import _LRScheduler

# AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


# class _BaseWarmupScheduler(_LRScheduler):

#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.successor = successor
#         self.warmup_epoch = warmup_epoch
#         super().__init__(optimizer, last_epoch, verbose)

#     def get_lr(self):
#         raise NotImplementedError

#     def step(self, epoch=None):
#         if self.last_epoch >= self.warmup_epoch:
#             self.successor.step(epoch)
#             self._last_lr = self.successor.get_last_lr()
#         else:
#             super().step(epoch)


# class ConstantWarmupScheduler(_BaseWarmupScheduler):

#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         cons_lr,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.cons_lr = cons_lr
#         super().__init__(
#             optimizer, successor, warmup_epoch, last_epoch, verbose
#         )

#     def get_lr(self):
#         if self.last_epoch >= self.warmup_epoch:
#             return self.successor.get_last_lr()
#         return [self.cons_lr for _ in self.base_lrs]


# class LinearWarmupScheduler(_BaseWarmupScheduler):

#     def __init__(
#         self,
#         optimizer,
#         successor,
#         warmup_epoch,
#         min_lr,
#         last_epoch=-1,
#         verbose=False
#     ):
#         self.min_lr = min_lr
#         super().__init__(
#             optimizer, successor, warmup_epoch, last_epoch, verbose
#         )

#     def get_lr(self):
#         if self.last_epoch >= self.warmup_epoch:
#             return self.successor.get_last_lr()
#         if self.last_epoch == 0:
#             return [self.min_lr for _ in self.base_lrs]
#         return [
#             lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
#         ]


# def build_lr_scheduler(optimizer, optim_cfg):
#     """A function wrapper for building a learning rate scheduler.

#     Args:
#         optimizer (Optimizer): an Optimizer.
#         optim_cfg (CfgNode): optimization config.
#     """
#     lr_scheduler = optim_cfg.LR_SCHEDULER
#     stepsize = optim_cfg.STEPSIZE
#     gamma = optim_cfg.GAMMA
#     max_epoch = optim_cfg.MAX_EPOCH

#     if lr_scheduler not in AVAI_SCHEDS:
#         raise ValueError(
#             f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
#         )

#     if lr_scheduler == "single_step":
#         if isinstance(stepsize, (list, tuple)):
#             stepsize = stepsize[-1]

#         if not isinstance(stepsize, int):
#             raise TypeError(
#                 "For single_step lr_scheduler, stepsize must "
#                 f"be an integer, but got {type(stepsize)}"
#             )

#         if stepsize <= 0:
#             stepsize = max_epoch

#         scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=stepsize, gamma=gamma
#         )

#     elif lr_scheduler == "multi_step":
#         if not isinstance(stepsize, (list, tuple)):
#             raise TypeError(
#                 "For multi_step lr_scheduler, stepsize must "
#                 f"be a list, but got {type(stepsize)}"
#             )

#         scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer, milestones=stepsize, gamma=gamma
#         )

#     elif lr_scheduler == "cosine":
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, float(max_epoch)
#         )

#     if optim_cfg.WARMUP_EPOCH > 0:
#         if not optim_cfg.WARMUP_RECOUNT:
#             scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

#         if optim_cfg.WARMUP_TYPE == "constant":
#             scheduler = ConstantWarmupScheduler(
#                 optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
#                 optim_cfg.WARMUP_CONS_LR
#             )

#         elif optim_cfg.WARMUP_TYPE == "linear":
#             scheduler = LinearWarmupScheduler(
#                 optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
#                 optim_cfg.WARMUP_MIN_LR
#             )

#         else:
#             raise ValueError

#     return scheduler


import torch
from torch.optim.lr_scheduler import _LRScheduler

# <--- MODIFIED: Added "cosine_with_restarts" to the list
AVAI_SCHEDS = ["single_step", "multi_step", "cosine", "cosine_with_restarts"]


class _BaseWarmupScheduler(_LRScheduler):
    # ... (此部分代码无需任何修改) ...
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    # ... (此部分代码无需任何修改) ...
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):
    # ... (此部分代码无需任何修改) ...
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
        
    # <--- ADDED: The new logic for cosine with restarts --->
    elif lr_scheduler == "cosine_with_restarts":
        # T_0 is the number of epochs for the first restart cycle.
        # This is a required parameter for this scheduler.
        # if "T_0" not in optim_cfg:
        #     raise ValueError("T_0 must be specified in the config for cosine_with_restarts.")
        
        T_0 = optim_cfg.get("T_0", 10)
        # T_mult is a factor to increase T_i after a restart. T_i = T_i * T_mult.
        # This is an optional parameter.
        T_mult = optim_cfg.get("T_MULT", 1)
        # eta_min is the minimum learning rate.
        eta_min = optim_cfg.get("ETA_MIN", 1e-7)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    # <--- END OF ADDED BLOCK --->

    if optim_cfg.WARMUP_EPOCH > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            # This logic might need adjustment if using restarts,
            # but for now we assume restarts happen after warmup.
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear":
            # Note: The original code uses WARMUP_MIN_LR, check your config
            # I will assume it should be WARMUP_CONS_LR based on your previous configs
            min_lr = optim_cfg.get("WARMUP_MIN_LR", optim_cfg.get("WARMUP_CONS_LR", 0))
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                min_lr
            )

        else:
            raise ValueError

    return scheduler
