"""Demonstration of HuggingFace, OCR, and ASCII ReplayStrategyMixins.

Usage:

    $ python -m openadapt.replay DemoReplayStrategy
"""

from loguru import logger

from openadapt.db import crud
from openadapt.models import Recording, Screenshot, WindowEvent
from openadapt.strategies.base import BaseReplayStrategy
from openadapt.strategies.mixins.ascii import ASCIIReplayStrategyMixin
from openadapt.strategies.mixins.huggingface import (
    MAX_INPUT_SIZE,
    HuggingFaceReplayStrategyMixin,
)
from openadapt.strategies.mixins.ocr import OCRReplayStrategyMixin
from openadapt.strategies.mixins.sam import SAMReplayStrategyMixin
from openadapt.strategies.mixins.summary import SummaryReplayStrategyMixin

import textgrad as tg
import numpy as np
import random
from textgrad.tasks import load_task

class DemoReplayStrategy(
    HuggingFaceReplayStrategyMixin,
    OCRReplayStrategyMixin,
    ASCIIReplayStrategyMixin,
    SAMReplayStrategyMixin,
    SummaryReplayStrategyMixin,
    BaseReplayStrategy,
):
    """Demo replay strategy that combines HuggingFace, OCR, and ASCII mixins."""

    def __init__(
        self,
        recording: Recording,
    ) -> None:
        """Initialize the DemoReplayStrategy.

        Args:
            recording (Recording): The recording to replay.
        """
        print("Initializing DemoReplayStrategy")
        super().__init__(recording)
        self.result_history = []
        session = crud.get_new_session(read_only=True)
        self.screenshots = crud.get_screenshots(session, recording)
        self.screenshot_idx = 0

        # Initialize TextGrad components
        print("Initializing TextGrad components")
        self.llm_api_eval = tg.get_engine(engine_name="gpt-4o")
        self.llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo-0125")
        tg.set_backward_engine(self.llm_api_eval, override=True)
        self.train_set, self.val_set, self.test_set, self.eval_fn = load_task("BBH_object_counting", evaluation_api=self.llm_api_eval)
        self.system_prompt = tg.Variable("", requires_grad=True, role_description="system prompt to the language model")
        self.optimizer = tg.TextualGradientDescent(engine=self.llm_api_eval, parameters=[self.system_prompt])
        self.results = {"test_acc": [], "prompt": [], "validation_acc": []}
        print("TextGrad components initialized")
        print("DemoReplayStrategy initialized")

    def get_next_action_event(
        self,
        screenshot: Screenshot,
        window_event: WindowEvent,
    ) -> None:
        """Get the next action event based on the current screenshot and window event.

        Args:
            screenshot (Screenshot): The current screenshot.
            window_event (WindowEvent): The current window event.

        Returns:
            None: No action event is returned in this demo strategy.
        """
        print("Executing get_next_action_event")
        screenshot_bbox = self.get_screenshot_bbox(screenshot)
        logger.info(f"screenshot_bbox=\n{screenshot_bbox}")

        screenshot_click_event_bbox = self.get_click_event_bbox(
            self.screenshots[self.screenshot_idx]
        )
        logger.info(
            "self.screenshots[self.screenshot_idx].action_event=\n"
            f"{screenshot_click_event_bbox}"
        )
        event_strs = [f"<{event}>" for event in self.recording.action_events]
        history_strs = [f"<{completion}>" for completion in self.result_history]
        prompt = " ".join(event_strs + history_strs)
        N = max(0, len(prompt) - MAX_INPUT_SIZE)
        prompt = prompt[N:]

        # Optimize the prompt using TextGrad
        self.system_prompt.set_value(prompt)
        print(f"Initial prompt: {prompt}")
        for epoch in range(3):
            print(f"Starting epoch {epoch}")
            for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(self.train_set, position=0))):
                print(f"Starting step {steps} of epoch {epoch}")
                pbar.set_description(f"Training step {steps}. Epoch {epoch}")
                self.optimizer.zero_grad()
                losses = []
                for (x, y) in zip(batch_x, batch_y):
                    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                    response = self.system_prompt(x)
                    try:
                        eval_output_variable = self.eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                    except:
                        eval_output_variable = self.eval_fn([x, y, response])
                    losses.append(eval_output_variable)
                total_loss = tg.sum(losses)
                print(f"Epoch {epoch}, Step {steps}, Loss: {total_loss.value}")
                total_loss.backward()
                self.optimizer.step()
                print(f"Completed step {steps} of epoch {epoch}")
                self.run_validation_revert()
            print(f"Completed epoch {epoch}")
        optimized_prompt = self.system_prompt.get_value()
        print(f"Optimized prompt: {optimized_prompt}")
        max_tokens = 10
        completion = self.get_completion(optimized_prompt, max_tokens)

        result = completion.split(">")[0].strip(" <>")
        self.result_history.append(result)

        self.screenshot_idx += 1
        print("Completed get_next_action_event")
        return None

    def run_validation_revert(self):
        print("Running validation revert")
        val_performance = np.mean(self.eval_dataset(self.val_set, self.eval_fn, self.system_prompt))
        previous_performance = np.mean(self.results["validation_acc"][-1])
        print("val_performance: ", val_performance)
        print("previous_performance: ", previous_performance)
        previous_prompt = self.results["prompt"][-1]

        if val_performance < previous_performance:
            print(f"rejected prompt: {self.system_prompt.value}")
            self.system_prompt.set_value(previous_prompt)
            val_performance = previous_performance

        self.results["validation_acc"].append(val_performance)
        print("Completed validation revert")

    def eval_dataset(self, test_set, eval_fn, model, max_samples: int=None):
        print("Evaluating dataset")
        if max_samples is None:
            max_samples = len(test_set)
        accuracy_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for _, sample in enumerate(test_set):
                future = executor.submit(self.eval_sample, sample, eval_fn, model)
                futures.append(future)
                if len(futures) >= max_samples:
                    break
            tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
            for future in tqdm_loader:
                acc_item = future.result()
                accuracy_list.append(acc_item)
                tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
        print("Completed dataset evaluation")
        return accuracy_list

    def eval_sample(self, item, eval_fn, model):
        print("Evaluating sample")
        x, y = item
        x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
        y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
        response = model(x)
        try:
            eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            return int(eval_output_variable.value)
        except:
            eval_output_variable = eval_fn([x, y, response])
            eval_output_parsed = eval_fn.parse_output(eval_output_variable)
            return int(eval_output_parsed)
