import static qupath.lib.scripting.QP.*
import static qupath.lib.gui.scripting.QPEx.*

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// No need to change
def model_path = "/workspace/paquo/models/cuda_mobilevit4.pt2"
def script_path = ""
def use_autocrop = "True" // "True" or "False"
// --------------------------------------------------------------------------
// Change these parameters as needed
def sampling_size = "384" // 224
def batch_size = "8"
def use_smoothing_algorithm = "True" // "True" or "False"
// def stride_ratio = "0.5"
// def pad_ratio = "0.2"
// --------------------------------------------------------------------------

if (use_smoothing_algorithm == "True") {
    script_path = "/workspace/paquo/scripts/smooth/run.py"
}
else {
    script_path = "/workspace/paquo/scripts/by_patch/run.py"
}

def project = getProject()
def command = ["python", "-u", script_path, "--project", project.path.toString(), "--model", model_path, "--sampling-patch-size", sampling_size, "--batch-size", batch_size, "--use_autocrop", use_autocrop]
def processBuilder = new ProcessBuilder(command)

processBuilder.redirectErrorStream(true)
def process = processBuilder.start()

def reader = new BufferedReader(new InputStreamReader(process.getInputStream()))
def line = ""
while ((line = reader.readLine()) != null) {
    print line + "\n"
}

def exitCode = process.waitFor()
if (exitCode == 0)
    print "[OK] Process completed successfully\n"
else 
    print "[FAILED] Process exited with code: $exitCode\n"
