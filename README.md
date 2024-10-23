
# **Context-Aware Prompt Tuning (CPT) - Demonstration Repo**

This repository demonstrates **Context-Aware Prompt Tuning (CPT)**, a method described in the paper [*Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods*](https://arxiv.org/abs/2410.17222). CPT combines the benefits of In-Context Learning (ICL), Prompt Tuning (PT), and adversarial techniques to achieve superior performance in few-shot learning scenarios.

The repo includes:
- An interactive **Jupyter notebook** (`demo.ipynb`) demonstrating how to use CPT with the **SST-2 dataset**.
- Integration with the **PEFT library**, allowing seamless extension and customization of CPT-based models.
- Instructions for setting up, training, and evaluating CPT models.

---

## **Table of Contents**
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Example Output](#example-output)
- [Citation](#citation)
- [License](#license)

---

## **Overview**  
Context-Aware Prompt Tuning (CPT) builds upon the principles of **In-Context Learning (ICL)** and **Prompt Tuning (PT)**, combining them with techniques inspired by **adversarial optimization** to improve the performance of large language models on **classification tasks**.

CPT works by:
- **Concatenating task-specific examples as context** before the input, leveraging ICL's strength of adapting to new tasks without modifying model parameters.
- **Optimizing the context embeddings iteratively**, allowing the model to better capture information from the provided examples, similar to PT.
- **Using projected gradient descent** to keep the updated embeddings close to their original values, ensuring that optimization remains stable and overfitting is minimized.

CPT achieves **better generalization** and **reduced overfitting** compared to traditional fine-tuning methods, with evaluations on multiple classification datasets demonstrating its effectiveness.

---

## **Setup Instructions**
You can run this project **locally** or **on Google Colab**. Follow the instructions below based on your preferred environment.

### Option 1: Running on Google Colab

Open the notebook directly in Colab using the link
 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UhQDVhZ9bDlSk1551SuJV8tIUmlIayta?usp=sharing)



### Option 2: Running Locally

1. **Clone the Repository:**
   ```bash
   !https://github.com/tsachiblau/CPT.git CPT_demo
   cd CPT_demo
   ```

2. **Install Dependencies:**
   Use the provided `requirements.txt` file to install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and Configure PEFT Library:**
   This repo integrates with a customized version of the **PEFT library**:
   ```bash
   git clone https://github.com/tsachiblau/peft_CPT.git peft
   pip install -e ./peft
   ```

4. **Launch Jupyter Notebook:**
   Open the `demo.ipynb` notebook in Jupyter or any other notebook interface:
   ```bash
   jupyter notebook notebooks/demo.ipynb
   ```
   


---

## **Repository Structure**

```
peft_CPT_Demo/
│
├── README.md         # Overview and instructions
├── LICENSE           # License (MIT or other)
├── requirements.txt  # List of dependencies
├── notebooks/
│   └── demo.ipynb    # Jupyter notebook with the CPT demonstration
├── peft/             # Fork of the PEFT library for CPT
└── .gitignore        # Ignoring unnecessary files (e.g., model cache)
```

---

## **Usage**

1. **Train the CPT Model:**
   The notebook walks through tokenizing, building the dataset, and training the CPT model. Key steps include:
   - Loading the **SST-2 dataset**.
   - Tokenizing the data and creating a **CPT-compatible dataset**.
   - Configuring and training the model using **Hugging Face's Trainer API**.

2. **Evaluate the Model:**
   The notebook also demonstrates how to generate predictions using the trained CPT model.

---

## **Example Output**

Sample output from the notebook after training and evaluation:

```
Input: input: The movie was boring output: negative
Prediction: negative
GT: negative

Input: input: I loved the plot and characters output: positive
Prediction: positive
GT: positive
```

---

## **Citation**

If you use this code or method in your research, please cite the original paper:

```
@inproceedings{blau2025cpt, title={Context-Aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods}, author={Tsachi Blau, Moshe Kimhi, Yonatan Belinkov, Alexander Bronstein, Chaim Baskin}, journal={arXiv preprint arXiv:2410.17222}}, year={2025} }
```

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
