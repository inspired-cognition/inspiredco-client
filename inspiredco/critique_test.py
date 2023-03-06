from unittest import mock

import pytest
import pytest_mock
import requests

import inspiredco.critique as critique


def test_evaluate_success(
    mocker: pytest_mock.MockerFixture,
):
    expected_result = {
        "overall": {"value": 0.5},
        "example": [{"value": 0.4}, {"value": 0.6}],
    }
    mocker.patch(
        "inspiredco.critique.Critique.submit_task",
        return_value="__DUMMY_TASK_ID__",
    )
    mocker.patch(
        "inspiredco.critique.Critique.fetch_status",
        return_value={"status": "succeeded"},
    )
    mocker.patch(
        "inspiredco.critique.Critique.fetch_result",
        return_value=expected_result,
    )
    critique_client = critique.Critique(api_key="__DUMMY_API_KEY__")
    actual_result = critique_client.evaluate(
        metric="__DUMMY_METRIC__",
        config={},
        dataset=[{"target": "dummy1"}, {"target": "dummy2"}],
    )
    assert expected_result == actual_result


def test_evaluate_with_parallel(
    mocker: pytest_mock.MockerFixture,
):
    expected_result = {
        "overall": {"value": 0.5},
        "example": [{"value": 0.5}] * 1000,
    }
    partial_result = {
        "overall": {"value": 0.5},
        "example": [{"value": 0.5}] * 250,  # 250 is the limit for BARTScore
    }
    mocker.patch(
        "inspiredco.critique.Critique.submit_task",
        return_value="__DUMMY_TASK_ID__",
    )
    mocker.patch(
        "inspiredco.critique.Critique.fetch_status",
        return_value={"status": "succeeded"},
    )
    mocker.patch(
        "inspiredco.critique.Critique.fetch_result",
        return_value=partial_result,
    )
    critique_client = critique.Critique(api_key="__DUMMY_API_KEY__")
    actual_result = critique_client.evaluate(
        metric="bart_score",
        config={},
        dataset=[{"target": "dummy"}] * 1000,
    )
    assert expected_result == actual_result


def test_wait_for_result_success(
    mocker: pytest_mock.MockerFixture,
):
    expected_result = {
        "overall": {"value": 0.5},
        "example": [{"value": 0.4}, {"value": 0.6}],
    }
    patched_fetch_status = mocker.patch(
        "inspiredco.critique.Critique.fetch_status",
        side_effect=[
            {"status": "queued"},
            {"status": "queued"},
            {"status": "succeeded"},
        ],
    )
    mocker.patch(
        "inspiredco.critique.Critique.fetch_result",
        return_value=expected_result,
    )
    critique_client = critique.Critique(api_key="__DUMMY_API_KEY__")
    actual_result = critique_client.wait_for_result(
        task_id="__DUMMY_TASK_ID__",
    )
    assert expected_result == actual_result
    fetch_status_calls = [mock.call("__DUMMY_TASK_ID__")] * 3
    patched_fetch_status.assert_has_calls(fetch_status_calls)


@pytest.mark.parametrize(
    "metric",
    ["bart_score", "bert_score", "comet", "detoxify", "uni_eval"],
)
def test_submit_task_size_limit(
    metric: str,
    mocker: pytest_mock.MockerFixture,
):
    metric_example_limits = {
        "bart_score": 250,
        "bert_score": 2000,
        "comet": 250,
        "detoxify": 2000,
        "uni_eval": 250,
    }
    dummy_response = requests.Response()
    dummy_response.status_code = 200
    dummy_response.encoding = "utf-8"
    dummy_response._content = b'{"task_id": "__DUMMY_TASK_ID__"}'
    mocker.patch(
        "inspiredco.critique.Critique.http_post",
        return_value=dummy_response,
    )
    critique_client = critique.Critique(api_key="__DUMMY_API_KEY__")
    # This should not raise an exception.
    good_dataset = [{"target": "dummy"}] * metric_example_limits[metric]
    actual_response = critique_client.submit_task(
        metric=metric, config={}, dataset=good_dataset
    )
    assert "__DUMMY_TASK_ID__" == actual_response
    # This should raise an exception due to the too-long input.
    bad_dataset = [{"target": "dummy"}] * (metric_example_limits[metric] + 1)
    with pytest.raises(ValueError) as e:
        critique_client.submit_task(metric=metric, config={}, dataset=bad_dataset)
    e.match(
        f"Metric {metric} is limited to {metric_example_limits[metric]} "
        f"examples, but {len(bad_dataset)} examples were provided."
    )


def test_submit_tasks_parallel(
    mocker: pytest_mock.MockerFixture,
):
    dummy_response = requests.Response()
    dummy_response.status_code = 200
    dummy_response.encoding = "utf-8"
    dummy_response._content = b'{"task_id": "__DUMMY_TASK_ID__"}'
    mocker.patch(
        "inspiredco.critique.Critique.http_post",
        return_value=dummy_response,
    )
    critique_client = critique.Critique(api_key="__DUMMY_API_KEY__")
    long_dataset = [{"target": "dummy"}] * 900
    expected_response = ["__DUMMY_TASK_ID__"] * 4
    actual_response = critique_client.submit_tasks_parallel(
        metric="bart_score",
        config={},
        dataset=long_dataset,
    )
    assert expected_response == actual_response


def test_merge_results():
    result1 = {"overall": {"value": 0.5}, "example": [{"value": 0.2}, {"value": 0.8}]}
    result2 = {"overall": {"value": 0.2}, "example": [{"value": 0.2}]}
    actual_result = critique.Critique.merge_results("bart_score", [result1, result2])
    assert actual_result["overall"]["value"] == pytest.approx(0.4)
    assert actual_result["example"][0]["value"] == pytest.approx(0.2)
    assert actual_result["example"][1]["value"] == pytest.approx(0.8)
    assert actual_result["example"][2]["value"] == pytest.approx(0.2)
