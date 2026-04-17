import os
from appworld.environment import AppWorld
from appworld.task import load_task_ids

task_ids = load_task_ids("train") + load_task_ids("dev")
if not task_ids:
    print("No tasks found.")
    exit(1)

task_id = task_ids[0]
print(f"Testing task: {task_id}")

with AppWorld(
    task_id=task_id,
    experiment_name="verification",
    remote_apis_url=None,
    remote_environment_url=None,
    remote_docker=False,
    raise_on_failure=True,
    ground_truth_mode="full",
) as world:
    code = world.task.ground_truth.compiled_solution_code
    code = code + "\nsolution(apis, requester)"
    print("Executing code:")
    print(code)
    world.execute(code)
    tracker = world.evaluate(suppress_errors=False)
    print("Success:", tracker.success)
    print("Is exception:", getattr(tracker, 'exception', 'No exception attribute'))
    if getattr(tracker, 'exception', None):
        print(tracker.exception)
    print("Evaluation details:")
    from pprint import pprint
    pprint(tracker.__dict__)
    if not tracker.success:
        print("Done debugging failure.")
