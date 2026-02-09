from __future__ import annotations as _annotations

import sys
from dataclasses import dataclass
from typing import Sequence

import pytest
from inline_snapshot import snapshot

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    ExperimentEvaluator,
    EvaluatorOutput,
)
from pydantic_evals.reporting import ReportCase

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.retries import RetryConfig
    from tenacity import stop_after_attempt

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.anyio,
]

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup

@dataclass
class PositiveSentiment(Evaluator[str, str, dict]):
    def evaluate(self, ctx: EvaluatorContext[str, str, dict]) -> bool:
        return 'happy' in ctx.output.lower()

@dataclass
class RowCount(ExperimentEvaluator[str, str, dict]):
    def evaluate_experiment(self, cases: Sequence[ReportCase[str, str, dict]]) -> int:
        return len(cases)

@dataclass
class SuccessRate(ExperimentEvaluator[str, str, dict]):
    def evaluate_experiment(self, cases: Sequence[ReportCase[str, str, dict]]) -> float:
        passing = sum(1 for c in cases if c.assertions.get('PositiveSentiment').value)
        return passing / len(cases) if cases else 0.0

async def mock_task(inputs: str) -> str:
    if inputs == 'case1':
        return 'I am happy'
    else:
        return 'I am sad'

async def test_experiment_evaluator_basic():
    """Test basic functionality of experiment-level evaluators with a simple success rate and row count."""
    dataset = Dataset(
        cases=[
            Case(name='case1', inputs='case1'),
            Case(name='case2', inputs='case2'),
        ],
        evaluators=[PositiveSentiment()],
        experiment_evaluators=[SuccessRate(), RowCount()]
    )

    report = await dataset.evaluate(mock_task, progress=False)

    assert report.experiment_scores['SuccessRate'].value == 0.5
    assert report.experiment_scores['RowCount'].value == 2
    assert report.experiment_scores['SuccessRate'].source.name == 'SuccessRate'

async def test_experiment_evaluator_async():
    """Test that async experiment evaluators are correctly awaited and executed."""
    @dataclass
    class AsyncRowCount(ExperimentEvaluator[str, str, dict]):
        async def evaluate_experiment(self, cases: Sequence[ReportCase[str, str, dict]]) -> int:
            return len(cases)

    dataset = Dataset(
        cases=[Case(name='case1', inputs='case1')],
        experiment_evaluators=[AsyncRowCount()]
    )

    report = await dataset.evaluate(mock_task, progress=False)
    assert report.experiment_scores['AsyncRowCount'].value == 1

async def test_experiment_evaluator_failure():
    """Test that failures in experiment evaluators are captured and reported without crashing the evaluation."""
    @dataclass
    class FailingExperimentEvaluator(ExperimentEvaluator[str, str, dict]):
        def evaluate_experiment(self, cases: Sequence[ReportCase[str, str, dict]]) -> EvaluatorOutput:
            raise ValueError('Experiment failure')

    dataset = Dataset(
        cases=[Case(name='case1', inputs='case1')],
        experiment_evaluators=[FailingExperimentEvaluator()]
    )

    report = await dataset.evaluate(mock_task, progress=False)
    assert len(report.experiment_evaluator_failures) == 1
    assert report.experiment_evaluator_failures[0].name == 'FailingExperimentEvaluator'
    assert 'ValueError: Experiment failure' in report.experiment_evaluator_failures[0].error_message

async def test_add_experiment_evaluator():
    """Test the add_experiment_evaluator method for adding evaluators to an existing dataset."""
    dataset = Dataset[str, str, dict](cases=[Case(name='case1', inputs='case1')])
    dataset.add_experiment_evaluator(RowCount())
    assert len(dataset.experiment_evaluators) == 1
    assert isinstance(dataset.experiment_evaluators[0], RowCount)

async def test_experiment_evaluator_serialization(tmp_path):
    """Test round-trip serialization and deserialization of datasets with experiment evaluators."""
    class MyDataset(Dataset[str, str, dict]):
        pass

    dataset = MyDataset(
        cases=[Case(name='case1', inputs='case1')],
        evaluators=[PositiveSentiment()],
        experiment_evaluators=[SuccessRate()]
    )

    yaml_path = tmp_path / 'dataset.yaml'
    dataset.to_file(yaml_path)

    loaded_dataset = MyDataset.from_file(
        yaml_path, 
        custom_evaluator_types=[PositiveSentiment, SuccessRate]
    )

    assert len(loaded_dataset.experiment_evaluators) == 1
    assert isinstance(loaded_dataset.experiment_evaluators[0], SuccessRate)

    report = await loaded_dataset.evaluate(mock_task, progress=False)
    assert report.experiment_scores['SuccessRate'].value == 1.0

@pytest.mark.skipif(not imports_successful(), reason='tenacity not installed')
async def test_experiment_evaluator_retry():
    """Test that experiment evaluators are retried according to the provided retry configuration."""
    attempts = 0

    @dataclass
    class RetryExperimentEvaluator(ExperimentEvaluator[str, str, dict]):
        def evaluate_experiment(self, cases: Sequence[ReportCase[str, str, dict]]) -> int:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError(f'Attempt {attempts} failed')
            return len(cases)

    dataset = Dataset(
        cases=[Case(name='case1', inputs='case1')],
        experiment_evaluators=[RetryExperimentEvaluator()]
    )

    report = await dataset.evaluate(
        mock_task,
        progress=False,
        retry_experiment_evaluators=RetryConfig(stop=stop_after_attempt(3))
    )

    assert attempts == 3
    assert report.experiment_scores['RetryExperimentEvaluator'].value == 1

async def test_experiment_evaluator_multiple_outputs():
    """Test that experiment evaluators can return multiple metrics of different types (score, label, assertion)."""
    @dataclass
    class MultiMetric(ExperimentEvaluator[str, str, dict]):
        def evaluate_experiment(self, cases: Sequence[ReportCase[str, str, dict]]) -> EvaluatorOutput:
            return {
                'count': len(cases),
                'status': 'finished',
                'is_valid': True
            }

    dataset = Dataset(
        cases=[Case(name='case1', inputs='case1')],
        experiment_evaluators=[MultiMetric()]
    )

    report = await dataset.evaluate(mock_task, progress=False)
    assert report.experiment_scores['count'].value == 1
    assert report.experiment_labels['status'].value == 'finished'
    assert report.experiment_assertions['is_valid'].value is True
