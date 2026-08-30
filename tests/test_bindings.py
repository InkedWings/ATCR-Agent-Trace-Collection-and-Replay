from agenttrace.replay.bindings import Bindings


def test_recursive_paths_commands_and_dynamic_result_mapping():
    bindings = Bindings(
        {"/old/workspace": "/new/workspace", "/tmp": "/replay/tmp"}
    )
    value = {
        "path": "/old/workspace/a.txt",
        "command": "cat /tmp/fetch.log > /old/workspace/out.txt",
        "nested": ["/tmp/other"],
    }
    assert bindings.rewrite(value) == {
        "path": "/new/workspace/a.txt",
        "command": "cat /replay/tmp/fetch.log > /new/workspace/out.txt",
        "nested": ["/replay/tmp/other"],
    }
    bindings.learn_from_results(
        {"details": {"fullOutputPath": "/tmp/fetch.log"}},
        {"details": {"fullOutputPath": "/new/result.log"}},
    )
    assert bindings.rewrite("read /tmp/fetch.log") == "read /new/result.log"
