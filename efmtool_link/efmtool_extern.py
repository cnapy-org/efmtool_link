import tempfile
import time
import os
import subprocess
import numpy
import psutil

# try to find a working java executable
_java_executable = 'java'
try:
    cp = subprocess.run([_java_executable, '-version'])
    if cp.returncode != 0:
        _java_executable = ''
except:
    _java_executable = ''
if _java_executable == '':
    _java_executable = os.path.join(os.environ.get('JAVA_HOME', ''), "bin", "java")
    try:
        cp = subprocess.run([_java_executable, '-version'])
        if cp.returncode != 0:
            _java_executable = ''
    except:
        _java_executable = ''
if _java_executable == '':
    import efmtool_link.efmtool_intern # just to find java executable via jpype
    _java_executable = os.path.join(str(efmtool_link.efmtool_intern.jSystem.getProperty("java.home")), "bin", "java")
# or comment out the above and set directly:
# _java_executable = r'E:\mpi\Anaconda3\envs\cnapy\Library\jre\bin\java' 
# _java_executable = r'C:\Program Files\AdoptOpenJDK\jdk-11.0.8.10-hotspot\bin\java.exe'

efmtool_jar = os.path.join(os.path.dirname(__file__), 'lib', 'metabolic-efm-all.jar')
default_jvm_memory = max(psutil.virtual_memory().total//(1024**2) - 2048, 128) # MB
cpu_cores = psutil.cpu_count(False)
if cpu_cores == None:
    cpu_cores = os.cpu_count()
default_threads = min(6, cpu_cores) # limit to 6 threads because EFMtool does not appear to profit from more

def calculate_flux_modes(st : numpy.array, reversible, reaction_names=None, metabolite_names=None, java_executable=None,
                         return_work_dir_only=False, jvm_max_memory=default_jvm_memory, max_threads=default_threads,
                         print_progress_function=print, abort_callback=None):
    if java_executable is None:
        java_executable = _java_executable
    if reaction_names is None:
        reaction_names = ['R'+str(i) for i in range(st.shape[1])]
    if metabolite_names is None:
        metabolite_names = ['M'+str(i) for i in range(st.shape[0])]
    
    curr_dir = os.getcwd()
    work_dir = tempfile.TemporaryDirectory()
    os.chdir(work_dir.name)
    write_efmtool_input(st, reversible, reaction_names, metabolite_names)

    try:
        java_call = [java_executable, '-Xmx'+str(jvm_max_memory)+'M',
        "-cp", efmtool_jar, "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
        '-kind', 'stoichiometry', '-arithmetic', 'double', '-zero', '1e-10',
        '-compression', 'default', '-level', 'INFO', '-maxthreads', str(max_threads),
        '-normalize', 'min', '-adjacency-method', 'pattern-tree-minzero',
        '-rowordering', 'MostZerosOrAbsLexMin', '-tmpdir', '.', '-stoich', 'stoich.txt', '-rev',
        'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'binary-doubles', 'efms.bin']
        # 'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'matlab', 'efms.mat'],
        if abort_callback is None:
            java_call = java_call + ['-log', 'console']
            cp = subprocess.Popen(java_call, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
                                  universal_newlines=True)

            # might there be a danger of deadlock in case an error produces a large text output that blocks the pipe?
            while cp.poll() is None:
                ln = cp.stdout.readlines(1) # blocks until one line has been read
                if len(ln) > 0: # suppress empty lines that can occur in case of external termination
                    print_progress_function(ln[0].rstrip())
            print_progress_function("".join(cp.stderr.readlines()))
        else:
            java_call = java_call + ['-log', 'file', 'log.txt']
            cp = subprocess.Popen(java_call)
            while cp.poll() is None: # wait for log file to appear
                time.sleep(0.1)
                if os.path.isfile('log.txt'):
                    break
            with open("log.txt") as log_file:
                while cp.poll() is None:
                    time.sleep(1.0)
                    ln= log_file.readlines()
                    if len(ln) > 0:
                        print_progress_function("".join(ln).rstrip())
                    if abort_callback():
                        cp.kill()
                        time.sleep(1.0) # allow some time for EFMtool to die properly so that work_dir can be cleanly deleted
                        break
        success = cp.poll() == 0
    except:
        success = False
    os.chdir(curr_dir)
    if success:
        if return_work_dir_only:
            return work_dir
        else:
        # efms = read_efms_from_mat(work_dir)
            return read_efms_from_bin(os.path.join(work_dir.name, 'efms.bin'))
    else:
        print_progress_function("EFMtool failure")
        return None

def write_efmtool_input(st, reversible, reaction_names, metabolite_names):
    if type(st) is not numpy.ndarray: # in case st is a sparse array
        st = st.toarray()
    numpy.savetxt(r"stoich.txt", numpy.array(st))
    with open('revs.txt', 'w') as file:
        file.write(' '.join(str(x) for x in reversible))
    with open('mnames.txt', 'w') as file:
        file.write(' '.join('"' + x + '"' for x in metabolite_names))
    with open('rnames.txt', 'w') as file:
        file.write(' '.join('"' + x + '"' for x in reaction_names))

# loading can sometimes fail because of unclear string encoding used by MatFileWriter (e.g. when there is an 'ä' in the string)
# def read_efms_from_mat(folder : str) -> numpy.array:
#     # taken from https://gitlab.com/csb.ethz/efmtool/
#     # efmtool stores the computed EFMs in one or more .mat files. This function
#     # finds them and loads them into a single numpy array.
#     efm_parts : List[np.array] = []
#     files_list = sorted(glob.glob(os.path.join(folder, 'efms_*.mat')))
#     for f in files_list:
#         mat = scipy.io.loadmat(f, verify_compressed_data_integrity=False)
#         efm_parts.append(mat['mnet']['efms'][0, 0])

#     return numpy.concatenate(efm_parts, axis=1)

def read_efms_from_bin(binary_doubles_file : str) -> numpy.array:
    with open(binary_doubles_file, 'rb') as fh:
        num_efm = numpy.fromfile(fh, dtype='>i8', count=1)[0]
        num_reac = numpy.fromfile(fh, dtype='>i4', count=1)[0]
        numpy.fromfile(fh, numpy.byte, count=1) # skip binary flag (boolean written as byte)
        efms = numpy.fromfile(fh, dtype='>d', count=num_reac*num_efm)
    return efms.reshape((num_reac, num_efm), order='F')
