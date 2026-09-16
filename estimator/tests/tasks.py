from estimator.tasks import EstimatorTask, ProgressTracker, prepare_yaml_dataset, DataPrepareArgs, SimpleExecutor

task = EstimatorTask(id = "task_X", training_data_path="data/dolly_smoke", validation_path="data/dolly_smoke", estimator_name='krr_estimator')


executer = SimpleExecutor("WTF")
executer.execute(task)
# executer = SimpleExecutor.continue_session("executed_tasks/WTF/metadata.json")
