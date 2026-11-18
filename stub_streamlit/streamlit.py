"""Streamlit giả: chỉ để main_launcher.py import được trên máy chủ web (không dựng giao diện)."""
import functools


class _Dummy:
    def __getattr__(self, name):
        return _Dummy()

    def __call__(self, *a, **k):
        return _Dummy()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


def cache_resource(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    return lambda f: f


cache_data = cache_resource
session_state = _State()
sidebar = _Dummy()


def __getattr__(name):
    return _Dummy()
