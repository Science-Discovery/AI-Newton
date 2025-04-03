# AI-Newton

This is a proof-of-concept implementation of AI-Newton, a concept-driven discovery system that can formulate general physical laws in an unsupervised manner without any prior physical knowledge. AI-Newton integrates  an autonomous discovery workflow, powered by a knowledge base (KB) and knowledge representation. The KB is responsible for storing and managing structured knowledge, including symbolic concepts, specific laws, and general laws. The knowledge representation is a physical domain specific language (DSL) that allows for the representation of physical concepts and laws in a structured and formalized manner. Given a collection of physical experiments, AI-Newton can formulate symbolic general laws applicable across a wide problem scope without neither supervision nor any prior physical knowledge. As a proof-of-concept implementation, it can rediscover Newton's second law, law of gravitation, conservation laws and others in classical mechanics.

## Installation
The program currently only supports Linux. CUDA is also currently required."

**Pre-requisites:**
```
Python 3.11.9 or higher
Rust 1.84.0
Maple 2024
```
Conda is strongly recommended for managing an independent python environment for AI-Newton.

**Installation via Conda and Cargo:**

1. Clone the repository:
```
git clone https://github.com/Science-Discovery/AI-Newton.git
cd AI-Newton
```
2. Create a virtual environment and install the dependencies:
```
conda create -n ainewton python=3.11
conda activate ainewton
pip install -r requirements.txt
```
3. Compile Rust libraries:
```
cargo build --release
cp target/release/libcore.so aiphy/core.so
```
4. Run the bench-marking test:
You can run `aiphy/test_human_operating-benchmark.ipynb` to test the installation. It will take about 10-20 minutes to run. 
5. Run the test case:
Since it may take several days to run the test case, it is recommended to run in the background:
```
nohup python -m test.examples.test_example_1 > logs/test_example_1/test_example_1.log 2>&1 &
```
Some of our test results can be found in the `data/test_cases/example_1` directory while logs can be found in the `data/test_cases/logs` directory. In which the *_knowledge.txt files are the most important. They are used to store the core part of the KB, including physical concepts and general laws discovered.


## Citation
