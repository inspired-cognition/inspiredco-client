from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from inspiredco import client_base

origin = "https://critique.api.inspiredco.ai"


class CritiqueStatus(TypedDict):
    metric: str
    status: Literal["queued", "succeeded", "failed"]
    detail: str | None


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
            RuntimeError: If the input values are invalid.
        """
        request_body = {"metric": metric, "config": config, "dataset": dataset}
        response = self.http_post(origin + "/evaluate/submit_task", request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Error {response.status_code} in submitting task: {response.text}."
            )
        return response.json()["task_id"]

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

            The "detail" key may include additional information.

        Raises:
            RuntimeError: If an error occurs.
        """
        request_body = {"task_id": task_id}
        response = self.http_post(origin + "/evaluate/fetch_status", request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Error {response.status_code} in fetching task {task_id} "
                f"status: {response.text}."
            )
        json_response = response.json()
        return {
            "metric": json_response["metric"],
            "status": json_response["status"],
            "detail": json_response.get("detail"),
        }

    def fetch_result(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """Get the result of a Critique task.

        Args:
            task_id: The ID of the task to fetch.

        Returns:
            The result of the task in dictionary format. This will include at
            least the following keys:
            * "overall": The overall score over the whole dataset.
            * "example": The score for each example in the dataset.

        Raises:
            RuntimeError: If an error occurs.
        """
        request_body = {"task_id": task_id}
        response = self.http_post(origin + "/evaluate/fetch_result", request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Error {response.status_code} in fetching task {task_id} "
                f"result: {response.text}."
            )
        return response.json()

    def wait_for_result(
        self,
        task_id: str,
    ) -> dict[str, Any]:
        """Wait for a task to finish and return the result.

        Args:
            task_id: The ID of the task to fetch.

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
                raise RuntimeError(f"Task failed: {full_status['detail']}")
            sleep_time = 1 if total_time < 10 else 5
            time.sleep(sleep_time)

        return self.fetch_result(task_id)

    def evaluate(
        self,
        metric: str,
        config: dict[str, Any],
        dataset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Make a request to Critique and wait for the result.

        Args:
            metric: The name of the metric to use.
            config: The configuration for the metric.
            dataset: The dataset to process.

        Returns:
            The result of the task in dictionary format. This will include at
            least the following keys:
            * "overall": The overall score over the whole dataset.
            * "example": The score for each example in the dataset.

        Raises:
            RuntimeError: If an error occurs.
        """
        # Submit task
        self._logger.info(f"Submitting task to {metric}.")
        task_id = self.submit_task(metric, config, dataset)
        return self.wait_for_result(task_id)
