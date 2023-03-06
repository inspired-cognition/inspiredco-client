from __future__ import annotations

import datetime
import statistics
import time
from typing import Any, Literal, TypedDict

from inspiredco import client_base
from inspiredco.critique_utils import exceptions

origin = "https://critique.api.inspiredco.ai"


metric_example_limits = {
    "bart_score": 250,
    "bert_score": 2000,
    "comet": 250,
    "detoxify": 2000,
    "uni_eval": 250,
}
unparalellizable_metrics = {"bleu"}


class CritiqueStatus(TypedDict):
    metric: str
    status: Literal["queued", "succeeded", "failed"]
    detail: str | None
    created_at: datetime.datetime | None
    updated_at: datetime.datetime | None


class Critique(client_base.ClientBase):
    """A client for Critique."""

    def submit_task(
        self,
        metric: str,
        config: dict[str, Any],
        dataset: list[dict[str, Any]],
    ) -> str:
        """Submit a task to Critique.

        Args:
            metric: The name of the metric to use.
            config: The configuration for the metric.
            dataset: The dataset to process.

        Returns:
            The task ID.

        Raises:
            CritiqueError: If the input values are invalid.
        """
        if metric in metric_example_limits:
            if len(dataset) > metric_example_limits[metric]:
                raise ValueError(
                    f"Metric {metric} is limited to {metric_example_limits[metric]} "
                    f"examples, but {len(dataset)} examples were provided."
                )
        request_body = {"metric": metric, "config": config, "dataset": dataset}
        response = self.http_post(origin + "/evaluate/submit_task", request_body)
        if response.status_code != 200:
            raise exceptions.RequestFailed(
                f"Error {response.status_code} in submitting task: {response.text}."
            )
        return response.json()["task_id"]

    def submit_tasks_parallel(
        self,
        metric: str,
        config: dict[str, Any],
        dataset: list[dict[str, Any]],
    ) -> list[str]:
        """Submit tasks to Critique in parallel.

        This is done by splitting up the task into chunks based on the metric's
        example limit. If the metric is not parallelizable, this will raise an
        error.

        Args:
            metric: The name of the metric to use.
            config: The configuration for the metric.
            dataset: The dataset to process.

        Returns:
            A list of task IDs for the submitted tasks.

        Raises:
            ValueError: If the input values are invalid.
        """
        if metric in unparalellizable_metrics:
            raise ValueError(f"Metric {metric} is not parallelizable.")
        example_limit = metric_example_limits.get(metric, len(dataset))
        task_ids = []
        for i in range(0, len(dataset), example_limit):
            task_ids.append(
                self.submit_task(
                    metric,
                    config,
                    dataset[i : i + example_limit],
                )
            )
        return task_ids

    def fetch_status(
        self,
        task_id: str,
    ) -> CritiqueStatus:
        """Fetch task status.

        Args:
            task_id: The ID of the task to fetch.

        Returns:
            The full status of the task as a dictionary.

            The "metric" key will include the name of the metric.

            The "status" key will be one of the following:
            * "queued": The task is queued and possibly processing.
            * "succeeded": The task has succeeded.
            * "failed": The task has failed.
            * "unknown": No data is available for that task ID.

            The "detail" key may include additional information.

        Raises:
            RuntimeError: If an error occurs.
        """
        request_body = {"task_id": task_id}
        response = self.http_post(origin + "/evaluate/fetch_status", request_body)
        if response.status_code != 200:
            raise exceptions.RequestFailed(
                f"Error {response.status_code} in fetching task {task_id} "
                f"status: {response.text}."
            )
        json_response = response.json()
        created_at = (
            datetime.datetime.fromisoformat(json_response["created_at"])
            if "created_at" in json_response
            else None
        )
        updated_str = json_response.get("updated_at")
        updated_at = (
            datetime.datetime.fromisoformat(updated_str) if updated_str else None
        )
        created_str = json_response.get("created_at")
        created_at = (
            datetime.datetime.fromisoformat(created_str) if created_str else None
        )
        return {
            "metric": json_response["metric"],
            "status": json_response["status"],
            "detail": json_response.get("detail"),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def fetch_result(
        self,
        task_id: str,
    ) -> dict[str, Any] | None:
        """Get the result of a Critique task.

        Args:
            task_id: The ID of the task to fetch.

        Returns:
            Either the result of the task in dictionary format, or None if there is
            no data available. The dictionary will include at least the following keys:
            * "overall": The overall score over the whole dataset.
            * "example": The score for each example in the dataset.

        Raises:
            RuntimeError: If an error occurs.
        """
        request_body = {"task_id": task_id}
        response = self.http_post(origin + "/evaluate/fetch_result", request_body)
        if response.status_code == 204:
            return None
        if response.status_code != 200:
            raise exceptions.RequestFailed(
                f"Error {response.status_code} in fetching task {task_id} "
                f"result: {response.text}."
            )
        return response.json()

    def wait_for_result(
        self,
        task_id: str,
        *,
        timeout: int = 600,
    ) -> dict[str, Any]:
        """Wait for a task to finish and return the result.

        Args:
            task_id: The ID of the task to fetch.
            timeout: The maximum time to wait for the task to finish in seconds.

        Returns:
            The result of the task in dictionary format.

        Raises:
            ValueError: If an error occurs.
        """
        # Wait
        start_time = time.time()
        while True:
            full_status = self.fetch_status(task_id)
            status = full_status["status"]
            # Get the time in seconds
            total_time = time.time() - start_time
            self._logger.info(f"Status {status} at {total_time:.1f}s.")
            if status == "succeeded":
                break
            if status == "failed":
                raise exceptions.TaskFailed(
                    f"Task {task_id} failed: {full_status['detail']}"
                )
            if total_time > timeout:
                raise exceptions.Timeout(
                    f"Task {task_id} did not complete in {timeout} seconds."
                )
            sleep_time = 1 if total_time < 10 else 5
            time.sleep(sleep_time)

        result = self.fetch_result(task_id)
        # Result should not be "none" if the task has succeeded
        assert result is not None
        return result

    @staticmethod
    def merge_results(
        metric: str,
        results: list[dict[str, Any]],
    ):
        """Merge the results of multiple tasks.

        Args:
            results: The results to merge.

        Returns:
            The merged results.

        Raises:
            ValueError: If the metric is not parallelizable.
        """
        if len(results) == 1:
            return results[0]
        if metric in unparalellizable_metrics:
            raise ValueError(f"Metric {metric} is not parallelizable.")
        examples: list[dict[str, Any]] = sum(
            (result["example"] for result in results), []
        )
        overall_keys = results[0]["overall"].keys()
        overall = {
            key: statistics.mean([example[key] for example in examples])
            for key in overall_keys
        }
        return {"overall": overall, "example": examples}

    def evaluate(
        self,
        metric: str,
        config: dict[str, Any],
        dataset: list[dict[str, Any]],
        timeout: int = 600,
    ) -> dict[str, Any]:
        """Make a request to Critique and wait for the result.

        Args:
            metric: The name of the metric to use.
            config: The configuration for the metric.
            dataset: The dataset to process.
            timeout: The maximum time to wait for the task to finish in seconds.

        Returns:
            The result of the task in dictionary format. This will include at
            least the following keys:
            * "overall": The overall score over the whole dataset.
            * "example": The score for each example in the dataset.

        Raises:
            RuntimeError: If an error occurs.
        """
        self._logger.info(f"Submitting task to {metric}.")
        if metric in unparalellizable_metrics:
            task_ids = [self.submit_task(metric, config, dataset)]
        else:
            task_ids = self.submit_tasks_parallel(metric, config, dataset)
        results = [
            self.wait_for_result(task_id, timeout=timeout) for task_id in task_ids
        ]
        return self.merge_results(metric, results)
