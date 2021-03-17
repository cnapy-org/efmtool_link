import tempfile
import glob
import os
import subprocess
import numpy
# import scipy

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

def calculate_flux_modes(st : numpy.array, reversible, reaction_names=None, metabolite_names=None, java_executable=None,
                         return_work_dir_only=False):
    if java_executable is None:
        java_executable = _java_executable
    if reaction_names is None:
        reaction_names = ['R'+str(i) for i in range(st.shape[1])]
    if metabolite_names is None:
        metabolite_names = ['M'+str(i) for i in range(st.shape[0])]
    
    curr_dir = os.getcwd()
#    with tempfile.TemporaryDirectory() as work_dir:
    work_dir = tempfile.TemporaryDirectory()
    print(work_dir.name)
    os.chdir(work_dir.name)
    write_efmtool_input(st, reversible, reaction_names, metabolite_names)

    try:
        cp = subprocess.Popen([java_executable,
        "-cp", efmtool_jar, "ch.javasoft.metabolic.efm.main.CalculateFluxModes",
        '-kind', 'stoichiometry', '-arithmetic', 'double', '-zero', '1e-10',
        '-compression', 'default', '-log', 'console', '-level', 'INFO',
        '-maxthreads', '-1', '-normalize', 'min', '-adjacency-method', 'pattern-tree-minzero', 
        '-rowordering', 'MostZerosOrAbsLexMin', '-tmpdir', '.', '-stoich', 'stoich.txt', '-rev', 
        'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'binary-doubles', 'efms.bin'],
        # 'revs.txt', '-meta', 'mnames.txt', '-reac', 'rnames.txt', '-out', 'matlab', 'efms.mat'],
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines=True)
        # might there be a danger of deadlock in case an error produces a large text output that blocks the pipe?
        while cp.poll() is None:
            ln = cp.stdout.readlines(1) # blocks until one line has been read
            if len(ln) > 0: # suppress empty lines that can occur in case of external termination
                print(ln[0], end='')
        print(cp.stderr.readlines())
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
        print("Emftool failure")
        return None

    # return efms

def write_efmtool_input(st, reversible, reaction_names, metabolite_names):
    numpy.savetxt(r"stoich.txt", st)
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
