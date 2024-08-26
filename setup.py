from setuptools import setup
import site
import os
from jpype._jvmfinder import getDefaultJVMPath, JVMNotFoundException, JVMNotSupportedException
from setuptools.command.install import install
import jdk

class PostInstallCommand(install):
    """Install a JRE if necessary."""
    
    def run(self):
        print("start PostInstallCommand run")
        try:
            getDefaultJVMPath()
        except (JVMNotFoundException, JVMNotSupportedException):
            paths = site.getsitepackages()
            paths.append(site.getusersitepackages())
            has_jre = False
            for path in paths:
                if os.access(path, os.W_OK):
                    path = os.path.join(path, 'jre')
                    if os.path.exists(path):
                        os.environ['JAVA_HOME'] = path
                        try:
                            getDefaultJVMPath()
                        except (JVMNotFoundException, JVMNotSupportedException):
                            pass
                        else:
                            print("Found existing Java Runtime Environtment in:", path)
                            has_jre = True
                            break
                    print("Installing Java Runtime Environtment in:")
                    print(jdk.install('11', jre=True, path=os.path.join(path, 'jre')))
                    has_jre = True
                    break
            if not has_jre: # very unlikely
                print("Could not install a Java Runtime Environtment, you need to install one yourself.")
        print("end PostInstallCommand run")

setup(name='efmtool_link',
      packages=['efmtool_link', 'efmtool_link.lib'],
      package_dir={'efmtool_link': 'efmtool_link'},
      package_data={'efmtool_link': ['lib/*.jar']},
      cmdclass={
          'install': PostInstallCommand,
      },
      zip_safe=False)
