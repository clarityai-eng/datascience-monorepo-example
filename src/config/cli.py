import argparse

from pydantic import BaseModel


def add_model(parser: argparse.ArgumentParser, model: BaseModel):
    """Add Pydantic model to an ArgumentParser

    Example:
        >>> from pydantic import BaseModel
        >>> class MyItem(BaseModel):
        ...     name: str
        ...     price: float
        ...     is_offer: bool = None
        >>> parser = argparse.ArgumentParser()
        >>> add_model(parser, MyItem)
        >>> args = parser.parse_args(["--name", "foo", "--price", "42.0"])
        >>> item = MyItem(**vars(args))
        >>> item
        MyItem(name='foo', price=42.0, is_offer=None)
    """
    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.type_,
            default=field.default,
            help=field.field_info.description,
        )
