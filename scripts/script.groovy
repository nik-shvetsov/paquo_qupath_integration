import static qupath.lib.scripting.QP.*
import static qupath.lib.gui.scripting.QPEx.*

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
def script_path = '/workspace/paquo/scripts/run.py'
def model_path = '/workspace/paquo/models/cuda_mobilevit4.pt2'
def sampling_size = '384'
// def batch_size = '8'
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------

def project = getProject()

// Define the command to run the Python script
def command = ["python", script_path, "--project", project.path.toString(), "--model", model_path, "--sampling-patch-size", sampling_size]

// Create a ProcessBuilder
def processBuilder = new ProcessBuilder(command)

// Redirect the error stream to standard output
processBuilder.redirectErrorStream(true)

// Start the process
def process = processBuilder.start()

// Get the output of the process
def reader = new BufferedReader(new InputStreamReader(process.getInputStream()))
def line = ""
while ((line = reader.readLine()) != null) {
    print line + "\n"
}

// Wait for the process to complete
def exitCode = process.waitFor()
if (exitCode == 0)
    print "[OK] Process completed successfully\n"
else 
    print "[FAILED] Process exited with code: $exitCode\n"
