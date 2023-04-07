from trainer.config.run import ConfigBase

from trainer.config.main import configclass
from trainer import Derived, Stateless
import typing as ty

from trainer.config.types import Dict, Enum, List, Literal, Optional, Tuple


@configclass
class TestConfig(ConfigBase):
    a1: int = 10


@configclass
class ParentTestConfig(ConfigBase):
    a1: int = 10
    c: TestConfig
    c2: TestConfig
    a2: str = 10


@configclass
class ParentTestTestConfig(ConfigBase):
    # TODO fix mypy issue
    a1: Derived[int] = 10
    c: ParentTestConfig


class myEnum(Enum):
    A = "a"


class Pass:
    def __init__(self, a) -> None:
        self.a = a


@configclass
class MultiTypeConfig(ConfigBase):
    # a0: Derived[Literal["a", "b", "2"]] = "a"
    # a1: int = 10
    # a2: int = 10
    a5: Pass

    a9: Derived[Dict[str]]
    a8: Derived[Optional[str]] = 10

    p1: Pass = Pass(a=10)
    p2: Pass = Pass(a="10")

    c2: TestConfig = TestConfig()
    c3: TestConfig
    c4: TestConfig
    a6: myEnum = "a"


@configclass
class ErrorConfigType(ConfigBase):
    a1: int = "2.2"


@configclass
class ErrorConfigHintOrder(ConfigBase):
    a4: Optional[Derived[str]] = "a"


@configclass
class ErrorConfigLiteral(ConfigBase):
    a0: Derived[Literal["a", "b", "2"]] = 10


@configclass
class ErrorConfigNonAnnotated(ConfigBase):
    a10 = 10


@configclass
class ErrorConfigBadAnnotated(ConfigBase):
    # Should throw an error for Optional[Derived]
    a4: ty.Dict[str, str]


@configclass
class ErrorConfigBadAnnotatedTwo(ConfigBase):
    # Should throw an error for Optional[Derived]
    a4: ty.Optional[str]


@configclass
class ErrorConfigEnum(ConfigBase):
    a4: myEnum = "b"


@configclass
class ErrorConfigTuple(ConfigBase):
    a4: Tuple[int, str] = ("2.1", "a")


@configclass
class ErrorConfigTupleLen(ConfigBase):
    a4: Tuple[int, str] = (10, "a", "a")


@configclass
class ErrorConfigList(ConfigBase):
    a4: List[str] = "a"




def test_types():
    e = MultiTypeConfig(a5={"a": 1}, c3={"a1": 2.4}, c4={"a1": "2"})
    assert e.a5.a == 1
    assert e.p1.a == 10
    assert e.p2.a == "10"
    assert e.a6 == "a"
    assert e.c3.a1 == 2
    assert e.c2.a1 == 10
    assert e.c4.a1 == 2
    assert e.a8 == "10"
    assert e.a9 is None
    try:
        e = MultiTypeConfig()
        assert False
    except Exception as excp:
        assert str(excp) == "Missing required value ['a5', 'c3', 'c4']"
    try:
        e = MultiTypeConfig(a5={"a": 1}, c3={"a1": 2.4}, c4={"a1": "2.2"})
        assert False
    except Exception as excp:
        assert str(excp) == "invalid literal for int() with base 10: '2.2'"



def test_error_configs():

    ERROR_CONFIGS = [
        (MultiTypeConfig, "Missing required value ['a5', 'c3', 'c4']"),
        (
            lambda: MultiTypeConfig(a5={"a": 1}, c3={"a1": 2.4}, c4={"a1": "2.2"}),
            "invalid literal for int() with base 10: '2.2'",
        ),
        (
            lambda: TestConfig("10"),
            "TestConfig does not support positional arguments.",
        ),
        (
            ErrorConfigTupleLen,
            "Incompatible lengths for a4 between (10, 'a', 'a') and type_hint: (<class 'int'>, <class 'str'>)",
        ),
        (ErrorConfigTuple, "invalid literal for int() with base 10: '2.1'"),
        (ErrorConfigEnum, "b is not supported by <enum 'myEnum'>"),
        (ErrorConfigLiteral, "10 is not a valid Literal ('a', 'b', '2')"),
        (ErrorConfigNonAnnotated, "All variables must be annotated. {'a10'}"),
        (
            ErrorConfigBadAnnotated,
            "Invalid collection <class 'dict'>. type_hints must be structured as:"
        ),
        (
            ErrorConfigBadAnnotatedTwo,
            "Invalid collection typing.Union. type_hints must be structured as:"
            # "Invalid collection <class 'trainer.config.types.Derived'>. type_hints must be structured as:",
        ),
        (
            ErrorConfigHintOrder,
            "Invalid collection <class 'trainer.config.types.Derived'>. type_hints must be structured as:",
        ),
        (ErrorConfigType, "invalid literal for int() with base 10: '2.2'"),
    ]
    for error_config, error_msg in ERROR_CONFIGS:
        try:
            error_config()
            assert False
        except Exception as excp:
            if not error_msg == str(excp):
                raise excp
    assert True
    pass

def test_hierarchical():

    c = TestConfig(a1="10")
    assert type(c.a1) == int and c.a1 == int("10")
    # Should fail
    # pc = ParentTestConfig(0,c,c,0)
    # Should not fail
    pc = ParentTestConfig(a1=0, c=c, c2=c)
    assert pc.a2 == str(10), "Could not cast"
    pc.c.a1 = 2
    assert pc.c2.a1 == pc.c.a1, "Lost reference"
    pc = ParentTestConfig(a1=0, c={"a1": 10}, c2={"a1": "2"}, a2=0)
    assert type(pc.c) == TestConfig
    assert pc.c2.a1 == 2
    pc_dict = ParentTestTestConfig(c=pc.to_dict())
    pc_obj = ParentTestTestConfig(c=pc)
    assert pc_dict == pc_obj


if __name__ == "__main__":
    pass
    test_types()
    test_hierarchical()
    test_error_configs()