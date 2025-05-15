If bu running the the train_mnist.py script you encounter an error like this

```
Traceback (most recent call last):
_.py", line 16, in <module>
    from .convert import convert_sklearn, to_onnx, wrap_as_onnx_mixin
  File "D:\repos\python\onnx-ml\.venv\Lib\site-packages\skl2onnx\convert.py", line 8, in <module>dule>
    from .proto import get_latest_tested_opset_version                                      , in <module>
  File "D:\repos\python\onnx-ml\.venv\Lib\site-packages\skl2onnx\proto\__init__.py", line 22, in <module>                                                                               \onnx-ml\.venv\Lib\site-packages\onnx\helper
    from onnx.helper import split_complex_to_pairs
ImportError: cannot import name 'split_complex_to_pairs' from 'onnx.helper' (D:\repos\python\onnx-ml\.venv\Lib\site-packages\onnx\helper.py)
```

Go to .venv/Lib/site-packages/skl2onnx/proto/__init__.py and **remove the following import**

```py
from onnx.helper import split_complex_to_pairs
```