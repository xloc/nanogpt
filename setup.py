from setuptools import setup

setup(
    name='nanogpt',
    version='0.1.0',
    packages=['nanogpt'],
    install_requires=[
        # requirements
        # 'requests == 2.22.0',
        # 'importlib; python_version == "2.6"',
        'tiktoken',
        'numpy',
        'torch',
        'datasets',
    ],
    # entry_points=dict(
    #     # <command>=<module>.<function_name>
    #     # more on https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    #     console_scripts="pdu=tptools.pdu:cli"
    # )
)
