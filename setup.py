from setuptools import setup

setup(name='efmtool_link',
      version='0.0.5',
      description='Exposes various efmtool functionality in Python.',
      url='https://github.com/cnapy-org/efmtool_link.git',
      author='Axel von Kamp',
      author_email='axelk1@gmx.de',
      license='Apache License 2.0',
      packages=['efmtool_link'],
      package_data={'efmtool_link': ['lib/*.jar']},
      install_requires=['jpype1', 'numpy', 'scipy', 'cobra', 'psutil'],
      zip_safe=False)
