from __future__ import annotations

from data_prep import load_data_prep_config, prepare_dataset_from_config
from typing import Any, Literal, Annotated
from pydantic import Field, BaseModel
from datetime import datetime
from pathlib import Path
import hashlib
import json
from mlx_lm import load
from mlops import load_lora_run_config, _deep_merge
from estimator.lora_trainer import SelectorConfig
from estimator.estimators import KernelArgs, Trainer, BaselineEstimator, KernelEstimator
from kernel.config import KernelMethodConfig
from kernel.data import PairRecord

TaskName = Literal["estimate_loss"]
TaskStatus = Literal["assigned", "failed", "finished","computing"]
SubsetStrategy = Literal["all", "ratio", "clipped_ratio", "fixed"]
EstimatorName = Literal["baseline_estimator", "krr_estimator"]
KernelMethodName = Literal["nystrom", "dual"]
FeatureTransformName = Literal["identity", "thresholded_sign"]

SUPPORTED_STRATEGIES = {"all", "ratio", "clipped_ratio", "fixed"}
ESTIMATOR_NAMES = {"baseline_estimator", "krr_estimator"}

class MlxArgs(BaseModel):
    lr : Annotated[float, Field(gt=0)] = 10e-5
    iters : Annotated[int, Field(gt=0)] = 1200
    max_seq_length : Annotated[int, Field(gt=0)] = 512
    batch_size : Annotated[int, Field(gt=0)] = 4
    grad_acum : Annotated[int, Field(gt=0)] = 4
    lora_rank : Annotated[int, Field(gt=0)] = 8
    lora_scale : Annotated[float, Field(gt=0)] = 20.0

class SubsetStrategyArgs(BaseModel):
    ratio : Annotated[float, Field(gt=0)] = 0.1
    max : Annotated[int, Field(gt=0)] = 512

def default_mlx_args() -> MlxArgs:
    return MlxArgs()

def default_subset_args() -> SubsetStrategyArgs:
    return SubsetStrategyArgs()

class EstimatorTask(BaseModel):
    id : Annotated[str, Field(min_length=1)]
    training_data_path : str
    name : TaskName = "estimate_loss"
    estimator_name : EstimatorName = "baseline_estimator"
    validation_path : str | None = None
    model : str = "mlx-community/SmolLM2-1.7B-Instruct"

    mlx_args : MlxArgs = Field(default_factory=default_mlx_args)
    feature_transform : FeatureTransformName = "identity"
    threshold : Annotated[float, Field(ge=0)] = 0.1

    kernel_method : KernelMethodName = "nystrom"
    ridge_lambda : Annotated[float, Field(gt=0)] = 0.01
    kernel_rank : Annotated[int, Field(gt=0)] = 32
    num_landmarks : Annotated[int, Field(gt=0)] = 128

    subset_strategy : SubsetStrategy = "clipped_ratio"
    subset_args : SubsetStrategyArgs = Field(default_factory = default_subset_args)
    seed : Annotated[int, Field(gt=0)] = 42

def save_plan(tasks: list[EstimatorTask], path : Path | str)->None:
    if not tasks:
        Warning("Trying to save an empty list of tasks. Ingnoring the command.")
        return

    payload = []
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        payload.append(task.model_dump_json())

    path.write_text('\n'.join(payload))


def load_plan(path: Path | str, default: list[EstimatorTask] | None = None) -> list[EstimatorTask]:
    abs_path = Path(path).resolve()
    if not abs_path.exists():
        if not default:
            raise FileNotFoundError(f"No plan found at {path}")
        return default

    str_tasks = abs_path.read_text().split('\n')
    tasks = []

    for task in str_tasks:
        try:
            x = EstimatorTask.model_validate_json(task)
        except Exception:
            Warning(f"Skipping a task at {path}")
            continue

        tasks.append(x)
    return tasks

def concat_task_lists(a : list[EstimatorTask], b : list[EstimatorTask]) -> list[EstimatorTask]:
    '''Returns a new list of tasks ab = [a,b]'''
    ab = []

    for x in a:
        y = x.model_copy()
        y.id = f"a{y.id}"
        ab.append(y)
    for x in b:
        y = x.model_copy()
        y.id = f"b{y.id}"
        ab.append(y)

    return ab


def concatinate(L : list[list[EstimatorTask]]) -> list[EstimatorTask]:
    '''Returns a new list of tasks ab = [a,b]'''
    if not L:
        return []
    
    taskList = []
    count = 0

    for l in L:
        for x in l:
            x_copy = x.model_copy()
            x_copy.id = f"{x_copy.id}_{count}"
            taskList.append(x_copy)
        count += 1

    return taskList


def extend_task_from_range(task: EstimatorTask, test_range: list[tuple[str,Any]]) -> list[EstimatorTask]:
    tests = []
    count = 0

    for k, v in test_range:
        dumps = task.model_dump()
        dumps[k] = v
        dumps['id'] = f"{dumps['id']}_{count}"

        try:
            est_tsk = EstimatorTask(**dumps)
        except TypeError:
            raise ValueError(f"Error occured when trying to assign field {k} with value {v}")
        tests.append(est_tsk)

        count+=1

    return tests


def extend_task(task: EstimatorTask, test_range: list[tuple[tuple[str,...] | str, Any]]) -> list[EstimatorTask]:
    tests = []
    count = 0

    for k, v in test_range:
        dumps = task.model_dump()
        dumps['id'] = f"{dumps['id']}_{count}"

        if isinstance(k, str):
            dumps[k] = v
        else:
            if not k or not v:
                raise ValueError(f"No values are provided for pair ({k},{v})")
            
            if len(k)==1:
                if not isinstance(v, tuple):
                    dumps[k[0]] = v
                elif len(v) == 1:
                    dumps[k[0]] = v[0]
                else:
                    raise ValueError("When a field is set with a tuple, value must be unique")
            else:
                if not isinstance(v, tuple) or len(v) != len(k):
                    raise ValueError("Inputs must be tuples of the same size.")
                for key, val in zip(k,v):
                    dumps[key] = val

        try:
            est_tsk = EstimatorTask(**dumps)
        except TypeError:
            raise ValueError(f"Error occured when trying to assign field {k} with value {v}")
        tests.append(est_tsk)

        count+=1

    return tests


def get_task_hash(task: EstimatorTask, ignore = {'id'}) -> str:
    task_copy = task.model_dump_json()
    ignored_dict = json.loads(task_copy)

    for ig in ignore:
        del ignored_dict[ig]

    return hashlib.sha256(json.dumps(ignored_dict).encode("utf-8")).hexdigest()

def deduplicate_task_list(listOfTasks: list[EstimatorTask], ignore = {'id'}) -> list[EstimatorTask]:
    seen_hashes = set()
    deduplicated = []

    for t in listOfTasks:
        hash = get_task_hash(t, ignore=ignore)
        if hash in seen_hashes:
            continue
        deduplicated.append(t)
        seen_hashes.add(hash)
    return deduplicated

def merge_task_lists(a: list[EstimatorTask], b: list[EstimatorTask], on={'id'}):
    return deduplicate_task_list(concat_task_lists(a, b), ignore=on)

def merge(L: list[list[EstimatorTask]], on={'id'}):
    if not L:
        return []

    merged = deduplicate_task_list(concatinate(L), ignore=on)
    
    return merged

Status = Literal['failed', 'completed', 'in_progress']

class LogEntry(BaseModel):
    id : Annotated[str, Field(min_length=1)]
    status : Status
    time : datetime = Field(default_factory=datetime.now)

class TaskSubmission(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    time_of_submission: datetime = Field(default_factory=datetime.now)
    code: int = 0
    report: dict[str, Any] | str | None = None

class ProgressTracker(BaseModel):
    in_progress : list[TaskSubmission] = Field(default_factory=list)
    completed : list[TaskSubmission] = Field(default_factory=list)
    failed : list[TaskSubmission] = Field(default_factory=list)

    log : list[LogEntry] = Field(default_factory=list)

    @property
    def completed_ids(self) -> list[str]:
        return [task.id for task in self.completed]

    @property
    def failed_ids(self) -> list[str]:
        return [task.id for task in self.failed]

    @property
    def in_progress_id(self) -> str | None:
        is_empty = False if self.in_progress else True
        return None if is_empty else self.in_progress[0].id

    @property
    def is_in_progress(self) -> bool:
        return True if self.in_progress_id else False

    @property
    def num_completed(self) -> int:
        return len(self.completed_ids)

    @property
    def num_failed(self) -> int:
        return len(self.failed_ids)

    @property
    def num_total(self) -> int:
        return len(self.log)

    @staticmethod
    def load(path: str | Path, default: ProgressTracker | None = None) -> ProgressTracker:
        full_path = Path(path).resolve()
        if not full_path.exists():
            if not default:
                raise FileNotFoundError(f"File at {path} not found")
            return default

        return ProgressTracker.model_validate_json(full_path.read_text())

    def save(self, path: str | Path) -> None:
        full_path = Path(path).resolve()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(self.model_dump_json())
    
    def submit_in_progress(self, id: str, code: int = 0, report: dict[str,Any] | str | None = None) -> None:
        ids_1 = [x.id for x in self.in_progress] + [x.id for x in self.completed]
        ids_2 = [x.id for x in self.failed]

        if id in ids_1:
            raise ValueError("ID is already submitted")
        
        new_sub = TaskSubmission(
                                id = id,
                                code = code,
                                report=report,
                                )

        if id in ids_2:
            for idx, val in enumerate(self.failed):
                if val.id == id:
                    self.failed.pop(idx)
        
        self.in_progress.append(new_sub)
        self.log.append(LogEntry(id = new_sub.id, status='in_progress'))

    def submit_completed(self, id: str, code: int = 0, report: dict[str,Any] | str | None = None) -> None:
        ids_1 = [x.id for x in self.completed] + [x.id for x in self.failed]
        ids_2 = [x.id for x in self.in_progress]

        if id in ids_1:
            raise ValueError("ID is already submitted")
        if not id in ids_2:
                raise ValueError("ID not found in submitted ID's")

        new_sub = TaskSubmission(
                                id = id,
                                code = code,
                                report=report,
                                )
        
        self.completed.append(new_sub)
        self.log.append(LogEntry(id = new_sub.id, status='completed'))

        for idx, val in enumerate(self.in_progress):
            if val.id == id:
                self.in_progress.pop(idx)

    def submit_failed(self, id: str, code: int = 0, report: dict[str,Any] | str | None = None) -> None:
        ids_1 = [x.id for x in self.completed] + [x.id for x in self.failed]
        ids_2 = [x.id for x in self.in_progress]

        if id in ids_1:
            raise ValueError("ID is already submitted")
        if not id in ids_2:
                raise ValueError("ID not found in submitted ID's")

        new_sub = TaskSubmission(
                                id = id,
                                code = code,
                                report=report,
                                )
        
        self.failed.append(new_sub)
        self.log.append(LogEntry(id = new_sub.id, status='failed'))

        for idx, val in enumerate(self.in_progress):
            if val.id == id:
                self.in_progress.pop(idx)


BASE_LORA_RUN_CONFIG = 'configs/base.yaml'
DEFAULT_SUBSET_POLICY = {
                            'iters_policy': 'match_full_exposure',
                            'min_iters': 1,
                        }

class DataPrepareArgs(BaseModel):
    path : str | Path
    tokenizer_model : str | None = None
    base_model : str | None = None
    output_dir : str | Path | None = None
    seed : Annotated[int, Field(ge=0)] | None = None

def prepare_yaml_dataset(args: DataPrepareArgs) -> None:
    def token_supervision_enabled() -> bool:
        filters = config.get("filters")
        if not isinstance(filters, dict):
            return False
        token_filter = filters.get("token_supervision")
        return isinstance(token_filter, dict) and bool(token_filter.get("enabled"))
    
    config = load_data_prep_config(args.path)
    tokenizer_model = args.tokenizer_model
    if args.base_model is not None and token_supervision_enabled():
        tokenizer_model = args.base_model
    counts = prepare_dataset_from_config(
        args.path,
        output_dir=args.output_dir,
        seed=args.seed,
        tokenizer_model=tokenizer_model,
    )
    dataset_name = config.get("dataset_name", "dataset")
    output_dir = args.output_dir or config.get("output_dir") or f"data/{dataset_name}"
    print(f"Wrote {dataset_name} splits to {output_dir}: {counts}")


DEFAULT_SAMPLE_SIZE = 1
DEFAULT_EXECUTER_PATH = 'executed_tasks'

class ExecutionError(Exception):
    def __init__(self, code: int, message: str) -> None:
        self.message = message
        self.code = code
        super().__init__(message)

class PartialResultError(ExecutionError):
    def __init__(self, code: int, message: str, payload: dict[str, Any] = dict()) -> None:
        self.payload = payload
        super().__init__(message=message, code=code)

class TaskManagementError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(code=104, message=message)

class DataValidationError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(code=105, message=message)

class InvalidMLXModel(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(code=106, message=message)

class DataLoadingError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(code=107, message=message)

class EstimatorTrainingError(ExecutionError):
    def __init__(self, message: str) -> None:
        super().__init__(code=200, message=message)

class EstimatorRunError(PartialResultError):
    def __init__(self, message: str, payload: dict[str, Any] = dict()) -> None:
        super().__init__(code=201, message=message, payload=payload)

class ExecuterMeta(BaseModel):
    session_name : Annotated[str, Field(min_length=1)]
    progress_manager : Path
    results : Path
    completed : Annotated[int, Field(ge=0)] = 0
    failed : Annotated[int, Field(ge=0)] = 0
    total_submitted : Annotated[int, Field(ge=0)] = 0
    queued : str | None = None
    plan : Path | None = None

class SimpleExecutor:
    def __init__(self, session_name: str = "experiment", exists_ok=False) -> None:
        self._session_name = session_name

        root_path = Path(DEFAULT_EXECUTER_PATH)/f'{session_name}'
        manager_path = root_path/'committed.json'
        self._metadata_path = root_path/'metadata.json'
        results_path = root_path/'results.jsonl'

        if root_path.exists() and not exists_ok:
            raise FileExistsError("Session already exists")

        self._metadata = ExecuterMeta(
                                        session_name=session_name,
                                        progress_manager=manager_path,
                                        results=results_path,
                                      ) 
        
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.write_text(self._metadata.model_dump_json())
        self._task_manager = ProgressTracker()

    @classmethod
    def continue_session(cls, session_root_path : str | Path):
        full_path = Path(session_root_path).resolve()

        if not full_path.exists():
            raise FileNotFoundError(f"Couldn't find the root file at {session_root_path}")
        if not str(session_root_path).endswith(".json"):
            raise ValueError(f"Root file must be a json file, but got {str(full_path).split('.')[-1]} instead")
        try:
            metadata = ExecuterMeta.model_validate_json(full_path.read_text())
        except Exception:
            raise ValueError(f"Couldn't load root file")
        
        simple_executer = cls(session_name=metadata.session_name, exists_ok=True)
        simple_executer._metadata = metadata
        simple_executer._task_manager = ProgressTracker.load(metadata.progress_manager)

        return simple_executer

    def _save_meta(self) -> None:
        self._metadata_path.write_text(self._metadata.model_dump_json())

    def _save_task_manager(self) -> None:
        self._task_manager.save(self._metadata.progress_manager)

    def execute(self, task : EstimatorTask) -> None:
        if self._task_manager.is_in_progress and self._task_manager.in_progress_id != task.id:
            raise TaskManagementError(f"Another task with id {self._task_manager.in_progress_id} is awaiting execution. Execute or abort it.")
        self._task_manager.submit_in_progress(id = task.id, code=0)
        self._save_meta()

        try:
            payload = self._execute_task(task)
            self._append_results(task_id=task.id, payload=payload)
            report = {'results':str(self._metadata.results)}
            self._task_manager.submit_completed(id = task.id, code = 1, report=report)
        except PartialResultError as e:
            self._append_results(task_id=task.id, payload=e.payload)
            report = {'partial_result' : str(self._metadata.results)}
            self._task_manager.submit_failed(id = task.id, code = e.code, report=report)
        except ExecutionError as e:
            self._task_manager.submit_failed(id=task.id, code = e.code, report=e.message)

        self._save_meta()
        self._save_task_manager()

    def abort(self, task_id : str) -> None:
        if self._task_manager.is_in_progress and task_id != self._task_manager.in_progress_id:
            raise TaskManagementError(f"Can't abort another task while task with id {self._task_manager.in_progress_id} is waiting")

        if not self._task_manager.is_in_progress:
            self._task_manager.submit_in_progress(id=task_id, code=0)

        self._task_manager.submit_failed(
                                            task_id, 
                                            code=-1,
                                            report="Aborted",
                                        )
        
        self._save_task_manager()

    def abort_in_progress(self) -> None:
        if self._task_manager.in_progress_id is not None:
            self.abort(self._task_manager.in_progress_id)

    def _append_results(self, task_id: str, payload: dict[str,Any]) -> None:
        with open(self._metadata.results, "a") as f:
            f.write(json.dumps({'task_id': task_id, 'result': payload}))

    @staticmethod
    def _load_pair_records(path: Path, split: str = "test") -> list[PairRecord]:
        records = []

        with open(path) as f:
            for idx, line in enumerate(f):
                if not line:
                    continue

                payload = json.loads(line)
                prompt = payload.get('prompt', '')
                completion = payload.get('completion', '')

                record = PairRecord(pair_id=f"{split}:{idx}",
                                    split=split,
                                    prompt=prompt,
                                    completion=completion)
                records.append(record)
        return records

    @staticmethod
    def _verify_jsonl_data(path: Path) -> int:
        count = 0
        with open(path) as f:
            for line in f:
                count += 1
                if not line:
                    continue
                payload = json.loads(line)

                if (not isinstance(payload, dict) or 
                    not isinstance(payload.get('prompt'),str) or
                    not isinstance(payload.get('completion'),str)
                ):
                    raise DataValidationError(f"Unsupported train instance format on line {count-1}. Expected prompt&completion.")
            
        return count

    @staticmethod
    def _configs_from_task(task: EstimatorTask) -> dict[str, Any]:
        def check_data(path : str | Path, data_split: str = "train") -> dict[str, Any]:
            if not Path(path).resolve().exists:
                raise DataValidationError(f"Data file at {path} not found")
            elif str(path).endswith(".yaml"):
                try:
                    config = load_data_prep_config(path)
                except Exception:
                    raise DataValidationError(f"Couldn't load yaml data configuration file from {path}")
                data_dir = config['output_dir']
                dataset_name = Path(data_dir).name
                data_type = "yaml"
            elif Path(path).resolve().is_dir():
                if not (Path(path).resolve()/f'{data_split}.jsonl').exists():
                    raise DataValidationError(f"Couldn't find train.jsonl at {path}")
                data_dir = path
                dataset_name = Path(data_dir).name
                data_type = "data directory"
            else:
                raise DataValidationError(f"Unknown data file formal '{str(path).split('.')[-1]}'")

            payload = {   
                        'name' : dataset_name,
                        'type' : data_type,
                        'dir' : data_dir,
                        }
            
            return payload
            
        estimator = task.estimator_name
        base_model = task.model
        train_data = task.training_data_path
        test_data = task.validation_path

        train_check_report = check_data(train_data)
  
        train_dataset_name = train_check_report['name']
        train_data_type = train_check_report['type']
        train_data_dir = train_check_report['dir']

        if test_data is not None:
            test_check_report = check_data(test_data, 'test')
            test_data_type = test_check_report['type']
        else:
            test_data_type = None

        try:
            load(base_model, lazy=True)
        except Exception as e:
            if str(e).find("404 Client Error") > -1:
                raise InvalidMLXModel(f"Model name '{base_model}' is unreachable with mlx-lm")
            else:
                raise InvalidMLXModel(f"An error occurred when trying to load the model={base_model}")

        base_lora_config = load_lora_run_config(BASE_LORA_RUN_CONFIG)
        base_lora_config.dataset_name = train_dataset_name
        base_lora_config.task = 'loss estimation'
        base_lora_config.data_dir = train_data_dir
        base_lora_config.run_tags = ['loss estimator']

        mlx_args = task.mlx_args.model_dump()

        sample_size = DEFAULT_SAMPLE_SIZE
        strategy = {'policy': task.subset_strategy, 'ratio':task.subset_args.ratio, 'max':task.subset_args.max}

        if task.subset_strategy == "fixed":
            sample_size = task.subset_args.max
            strategy = {'policy': 'fixed'}
   
        base_lora_config.mlx_args = _deep_merge(base_lora_config.mlx_args, mlx_args)
        base_lora_config.subset_training = DEFAULT_SUBSET_POLICY
        selector_args = SelectorConfig(max_examples=sample_size)

        if estimator == "baseline_estimator":
            payload =   {
                            "estimator" : "baseline",
                            "lora_run_config" : base_lora_config,
                            "selector_args" : selector_args,
                            "train_data_type" : train_data_type,
                            "test_data_type" : test_data_type,
                            "train_path" : Path(train_data)/'train.jsonl',
                            "test_path" : Path(test_data)/'test.jsonl' if test_data is not None else None,
                            "strategy" : strategy,
                        }
            return payload

        kernel_backend_args = {
                                'leaf_filter': 'lora_b_only',
                                'feature_transform': task.feature_transform,
                                'threshold': task.threshold,
                                }
        
        kernel = KernelMethodConfig(
                                        method=task.kernel_method,
                                        ridge_lambda=task.ridge_lambda,
                                        rank = task.kernel_rank,
                                        num_landmarks= task.num_landmarks,
                                    )
        
        kernel_args = KernelArgs(
                                    seed = task.seed,
                                    limit=sample_size,
                                    kernel=kernel,
                                    backend_args=kernel_backend_args,
                                )

        payload={
                    "estimator" : "krr",
                    "lora_run_config" : base_lora_config,
                    "selector_args" : selector_args,
                    "kernel_args" : kernel_args,
                    "train_data_type" : train_data_type,
                    "test_data_type" : test_data_type,
                    "train_path" : Path(train_data)/'train.jsonl',
                    "test_path" : Path(test_data)/'test.jsonl' if test_data is not None else None,
                    "strategy" : strategy,
                }

        return payload

    @staticmethod
    def _execute_task(task: EstimatorTask) -> dict[str,Any]:
        payload = SimpleExecutor._configs_from_task(task)
        
        train_type = payload['train_data_type']
        if train_type == 'yaml':
            try:
                prepare_dataset_from_config(payload['train_path'])
            except Exception:
                raise DataLoadingError(f"couldn't load train data from yaml configuration file ar {payload['train_path']}")

        test_type = payload['test_data_type']
        if test_type == 'yaml':
            try:
                prepare_dataset_from_config(payload['test_path'])
            except Exception:
                raise DataLoadingError(f"couldn't load test data from yaml configuration file ar {payload['test_path']}")
        if test_type is None:
            validation_records = []
        else:
            validation_records = SimpleExecutor._load_pair_records(payload['test_path'])

        dataset_size = SimpleExecutor._verify_jsonl_data(payload['train_path'])
        strategy = payload['strategy']
        strategy_name = strategy['policy']

        lora_config = payload['lora_run_config']
        selector_args = payload['selector_args']
        sample_size = selector_args.max_examples

        if strategy_name == "all":
            sample_size = dataset_size
        if strategy_name == "ratio":
            sample_size = int(dataset_size * strategy['ratio'])
        if strategy_name == "clipped_ratio":
            sample_size = min(int(dataset_size * strategy['ratio']), strategy['max'])
        if strategy_name == "fixed":
            sample_size = strategy['max']

        selector_args.max_examples = sample_size
        trainer = Trainer(lora_config=lora_config, selector_args=selector_args)
        estimator_name = payload['estimator']

        if estimator_name == "baseline":
            try:
                estimator_paths = trainer.train_baseline()
            except Exception:
                raise EstimatorTrainingError("Crashed during training the baseline estimator")   

            if payload['test_data_type'] is None:
                return_payload={"estimator_paths" : estimator_paths.model_dump_json()} 
                return return_payload

            estimator = BaselineEstimator(estimator_paths)
            try:
                loss = estimator(records=validation_records)
            except Exception:
                return_payload={"estimator_paths" : estimator_paths.model_dump_json()}
                raise EstimatorRunError(
                                                        message="Couldn't estimate validation loss after estimator training",
                                                        payload=return_payload,
                                                       )
            
        elif estimator_name == "krr":
            kernel_args = payload["kernel_args"]
            kernel_args.limit = sample_size

            try:
                estimator_paths = trainer.train_kernel(kernel_args)
            except Exception:
                raise EstimatorTrainingError("Crashed during kernel estimator training")

            if payload['test_data_type'] is None:
                partial_result = {"estimator_paths" : estimator_paths.model_dump_json()}
                return partial_result         

            estimator = KernelEstimator(estimator_paths)
            try:
                loss = estimator(records=validation_records)
            except Exception:
                partial_result = {"estimator_paths" : estimator_paths.model_dump_json()}
                raise EstimatorRunError(
                                            message="Couldn't estimate loss after kernel estimator training",
                                            payload=partial_result,
                                        )
        else:
            raise ValueError("Unknown estimator type")

        average = sum(loss) / len(loss) if len(loss) != 0 else 0.0

        payload={
                    'estimator_paths' : estimator_paths.model_dump_json(),
                    'loss_values' : [float(l) for l in loss],
                    'average' : float(average),
                }
        
        return payload 

