#  Mini Chatbot de Nutrição com RAG e Mistral-7B

Este projeto implementa um **Mini Chatbot baseado em Retrieval-Augmented Generation (RAG)** utilizando o modelo **Mistral-7B-Instruct-v0.2** e a biblioteca **LangChain**.
O chatbot responde a perguntas sobre **nutrição**, recuperando informações relevantes de uma base de documentos (`.txt` e `.pdf`) e gerando respostas **precisas e contextualizadas**.

---

## Objetivo

Criar um sistema que combine:

* **Recuperação de informações** com FAISS (vetor store)
* **Geração de respostas** com o modelo Mistral-7B

Exemplos de perguntas:

* “Quais alimentos são ricos em ferro?”
* “Como planejar uma dieta equilibrada?”

O tema **nutrição** foi escolhido pela sua relevância social e pela ampla disponibilidade de conteúdo confiável.

---

##  Tecnologias Utilizadas

| Tecnologia               | Função                                     |
| ------------------------ | ------------------------------------------ |
| Python 3.11              | Linguagem base do projeto                  |
| LangChain                | Framework para pipeline RAG                |
| Mistral-7B-Instruct-v0.2 | Modelo de linguagem local (formato GGUF)   |
| FAISS                    | Vetor store para busca semântica           |
| Sentence Transformers    | Geração de embeddings (`all-MiniLM-L6-v2`) |
| llama-cpp-python         | Execução local do modelo Mistral           |
| PyPDF2                   | Processamento de arquivos PDF              |

---

## Pré-requisitos

* **CPU moderna** (mín. 8 núcleos) ou **GPU NVIDIA** (para acelerar o modelo)
* **Memória**: mínimo 12 GB RAM (para modelos quantizados Q4)
* **Python**: 3.8 ou superior (recomenda-se 3.11)
* **Espaço em Disco**: \~5 GB para o modelo e documentos

---

##  Instalação

### 1. Criar Ambiente Virtual

```bash
conda create -n chatbot python=3.11
conda activate chatbot
```

### 2. Instalar Dependências

```bash
pip install -U langchain langchain-community langchain-huggingface pypdf faiss-cpu sentence-transformers tf-keras llama-cpp-python
```

**Com GPU NVIDIA** (opcional):

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

> 🔎 Substitua `cu122` pela sua versão de CUDA (`nvidia-smi`)

---

### 3. Baixar o Modelo Mistral-7B

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf \
--local-dir ./modelos --local-dir-use-symlinks False
```

---

### 4. Preparar os Documentos

```bash
mkdir -p data/nutricao
```

Adicione de **10 a 50 arquivos `.txt` ou `.pdf`** com conteúdos sobre nutrição:
*guias, artigos científicos, receitas saudáveis, materiais da OMS, etc.*

---

##  Configuração

### 🔗 Caminho do modelo

No script `chatbot_rag.py`, atualize:

```python
llm = LlamaCpp(
    model_path="./model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    ...
)
```

###  Ativando GPU (opcional)

```python
llm = LlamaCpp(
    model_path="./modelos/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    max_tokens=500,
    temperature=0.7,
    n_gpu_layers=40,
    verbose=True
)
```

---

##  Como Usar

Execute o script:

```bash
python chatbot_rag.py
```

Você verá no terminal:

```
 Bem-vindo ao Chatbot de Nutrição!
Digite sua pergunta ou 'sair' para encerrar.
```

Faça perguntas como:

* “Quais alimentos são ricos em ferro?”
* “Como planejar uma dieta rica em fibras?”

Para encerrar, digite `sair`.

---

##  Exemplo de Saída

```
Pergunta: Quais alimentos são ricos em ferro?

Resposta: Alimentos ricos em ferro incluem carne vermelha, fígado, espinafre, lentilhas, feijão e cereais fortificados. O ferro heme, encontrado em produtos animais, é mais facilmente absorvido pelo corpo.

Fontes: [{'source': 'data/nutricao/guia_nutrientes.pdf', 'page': 3}, {'source': 'data/nutricao/artigo_ferro.txt'}]

Pergunta: sair
Encerrando o chatbot...
```

---

##  Estrutura do Projeto

```
mini-chatbot-rag/
├── data/
│   └── nutricao/                   # Arquivos .txt e .pdf
├── modelos/
│   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf
├── chatbot_rag.py                 # Script principal
└── README.md
```

---

## Pipeline RAG

1. **Leitura dos documentos** (.txt/.pdf)
2. **Divisão em chunks** (\~1000 caracteres, com sobreposição de 200)
3. **Geração de embeddings** com `all-MiniLM-L6-v2`
4. **Armazenamento vetorial** no FAISS
5. **Busca semântica** com LangChain
6. **Resposta gerada** com Mistral-7B

---

##  Performance

* **Tempo de resposta (CPU)**: 5–25 segundos
* **Tempo de resposta (GPU)**: 2–6 segundos

### Dicas de otimização:

* Reduza `n_ctx` (ex: 1024) ou `max_tokens`
* Use quantizações menores (ex: Q2\_K)
* Use GPU se possível

---

##  Problemas Conhecidos

* **Erro ao sair**:

  ```
  TypeError: 'NoneType' object is not callable
  ```

  Bug conhecido do `llama-cpp-python`. Não afeta o uso. Pode ser mitigado liberando o modelo manualmente.

* **Baixo desempenho em PCs com <12 GB RAM**: use quantizações mais leves ou reduza `n_ctx`.

---

##  Contribuições

Este projeto foi desenvolvido para fins acadêmicos.

Sinta-se à vontade para sugerir:

* Suporte a outros modelos (ex.: LLaMA, Gemma)
* Interfaces gráficas com Gradio ou Streamlit
* Otimização para bases maiores

---

## 📄 Licença

Distribuído sob a **licença MIT**.
