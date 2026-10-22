import logging
import random
import time
from typing import Literal, Optional, Tuple, Union
from transformers import TrainerCallback
import torch
from transformers import DebertaV2ForSequenceClassification, PretrainedConfig
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np

DATA_TYPE_ENUM = Literal[
    "sfam_train",
    "sfam_val",
    "sfam_test",
    "lisa_train",
    "lisa_val",
    "lisa_test",
    "foreign_domain_test",
]

STEERING_TYPE_ENUM = Literal[
    "base",
    "prompt_group",
    "prompt_attribute",
    "prompt_group_attribute",
    "activation_base",
    "activation_group",
    "activation_attribute",
    "activation_group_attribute",
]

random.seed(42)
np.random.seed(42)
transformers.set_seed(42, deterministic=True)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)


def log_time(
    message: str,
    start_time: float,
    logger: logging.Logger,
    level: Literal["info", "debug"] = "info",
    remaining_time: float = -1.0,
    total_time: float = -1.0,
    extra: dict = {},
):
    full_message = f"{message} took {format_time(time.time() - start_time)}"
    extra["time"] = time.time() - start_time
    if remaining_time > 0:
        full_message += f" (remaining time: {format_time(remaining_time)})"
        # extra["remaining_time"] = remaining_time
    if total_time > 0:
        full_message += f" - total time {format_time(total_time)}"
        # extra["total_time"] = total_time
    if level == "info":
        logger.info(full_message, extra=extra)
    elif level == "debug":
        logger.debug(full_message, extra=extra)
    else:
        raise ValueError("Level must be either 'info' or 'debug'")


def format_time(seconds: float) -> str:
    days, remainder = divmod(
        seconds, 24 * 60 * 60
    )  # Separate days and remainder seconds
    days = int(days)
    time_str = time.strftime(
        "%H:%M:%S", time.gmtime(remainder)
    )  # Format remaining time

    if days > 0:
        return f"{days}-{time_str}"
    else:
        return time_str


class HyperparameterSearchLogger(TrainerCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        # Log the hyperparameters at the beginning of each trial.
        self.logger.info(
            f"Starting training with learning_rate {args.learning_rate} and weight_decay {args.weight_decay}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        # Log the hyperparameters and final metrics when a trial finishes.
        final_metrics = state.log_history[-1] if state.log_history else {}
        train_runtime = final_metrics.get("train_runtime", 0)
        log_time(
            message=f"Training with learning_rate {args.learning_rate} and weight_decay {args.weight_decay}",
            start_time=time.time() - train_runtime,
            logger=self.logger,
        )


class AutoModelWithSigmoid(DebertaV2ForSequenceClassification):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.sigmoid = torch.nn.Sigmoid()
        if config.problem_type == "regression":
            self.loss_fct = torch.nn.MSELoss()
        elif config.problem_type == "single_label_classification":
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(
                f"AutoModelWithSigmoid is not suitable for {config.problem_type}"
            )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output: SequenceClassifierOutput = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        loss = None
        if labels is not None:
            if isinstance(self.loss_fct, torch.nn.MSELoss):
                # Apply sigmoid function
                logits = self.sigmoid(output.logits)
            elif isinstance(self.loss_fct, torch.nn.BCEWithLogitsLoss):
                # Do not apply sigmoid function --> BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class
                logits = output.logits
                
            if self.num_labels == 1:
                loss = self.loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = self.loss_fct(logits, labels)
        
        # Apply sigmoid to restrict logits to range [0, 1]
        logits = self.sigmoid(output.logits)
        
        if not return_dict:
            output = (logits,) + output[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )