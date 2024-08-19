from setuptools import setup

setup(name='efmtool_link',
      packages=['efmtool_link'],
      package_dir={'efmtool_link': 'efmtool_link'},
      package_data={'efmtool_link': ['lib/*.jar']},
      zip_safe=False)
