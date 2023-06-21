# Compatibility Considerations

Deploying models through a monorepo has important compatibility considerations. There are several layers already in place to mitigate potential issues. However, the most crucial one, is being aware of why it is important and how it works.

## Challenges in Model Compatibility

Deploy a model to production means to save that model somewhere (MLFlow), retrieve it at runtime, and perform the inference. Saving the model means that we need to transform a in-memory Python object to long term storage. This process is called **serialization**. The most common way to serialize Python objects is with [Pickle](https://docs.python.org/3/library/pickle.html) (or the alternative [cloudpickle](https://github.com/cloudpipe/cloudpickle) used by MLFlow).

Serialization however, has some potential challenges. The most important one is the compatibility with the saved object with the version of Python and other packages. When we de-serialize an object with Pickle, it will reconstruct the in-memory representation of that object using the classes available.

### Example

Let's see it with an example. Let's define the following class:

```python
class MyClass:
    def __init__(self, init_var):
        self._var = init_var
    def get_var(self):
        return self._var
```

We can serialize it and reload an instance very simply like this:

```console
>>> import pickle
>>> instance_1 = MyClass("test")
>>> with open("instance.pkl", "wb+") as f:
...     pickle.dump(instance_1, f)
...
>>> with open("instance.pkl", "rb") as f:
...     loaded_instance = pickle.load(f)
...
>>> loaded_instance.get_var()
'test'
```

Now, let's change our class:

```python
class MyClass:
    def __init__(self, init_foo):
        self._foo = init_foo

    def get_foo(self):
        return self._foo
```

What happens if we try to reload the same pickled instance?

```console
>>> import pickle
>>> with open("instance.pkl", "rb") as f:
...     loaded_instance = pickle.load(f)
...
>>> loaded_instance.get_var()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'MyClass' object has no attribute 'get_var'
```

Even if the serialization succeed, we can see we don't have the methods as before. Or even worse:

```console
>>> loaded_instance.get_foo()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 5, in get_foo
AttributeError: 'MyClass' object has no attribute '_foo'
```

Even trying to use the instance as the new version intended, we run into errors because the internal attributes have changed, so the serialization can't rebuild the new instance structure.

The same issue can happen with third party packages and even with python standard libraries (bumping python version for example). If we saved an object with `scikit-learn<1.0` and we try to load it with `scikit-learn>=1.0`, there might be incompatibility issues that are not easy to resolve.

## How to Manage the Impact

Because this project works as a monorepo, we are in a high risk of having this compatibility issues when working with models trained in previous versions. Especially if the model uses common modules in the pipeline object (derived features transformers, imputation transformers...)

Luckily, there are easy methods to manage the impact of the that we already have in place

- **Testing + Backward compatibility**: with our current test suite we can detect if a change in a common module will break the contract with other modules ensuring backward compatibility
- **Retrain models for new code versions**: In this case, the code is compatible in the high level API, but we changed internal functions and attributes which can give us de-serialization errors. When a change like this is publish, we should trigger a retrain of the models so we have a pickled version in mlflow with the compatible code.
- **Docker version + model version**: for a given model version we should run the pipelines with the docker version it was trained on. This way the code is guaranteed to be compatible.

To summarize:

- Production pipelines won't be affected because docker versions are pinned.
- Compatibility is preserved with testing.
- Upgrade docker version means model retraining.
