from setuptools import setup

setup(
    name='pylabyk',
    version='0.0.1',
    packages=['pylabyk'],
    platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License 2.0',
        # 'Programming Language :: Python :: 2', # deprecated
        'Programming Language :: Python :: 3'
    ],
    install_requires=['torch>=1.0.0'],
    url='https://github.com/yulkang/pylabyk',
    license='Apache Software License 2.0',
    author='Yul HR Kang',
    author_email='hk2699@caa.columbia.edu',
    description="Pylab-style utilities; this library is meant for the author's personal use, but everyone is welcome to use it and contribute to it.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
