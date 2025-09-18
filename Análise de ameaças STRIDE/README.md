# Criação de uma aplicação para análise de riscos de segurança

O desafio da vez foi a criação de uma aplicação em Python que fizesse a análise de riscos de segurança, segundo as diretrizes STRIDE, com base em uma imagem de entrada. Como base para o projeto tomei o exemplo presente no repositório [Github STRIDE-demo da DIO](https://github.com/digitalinnovationone/stride-demo). O objetivo desa tarefa era, nominalmente:

Implementar, documentar e compartilhar um projeto que utilize Python, FastAPI e Azure OpenAI para criar uma API capaz de:

- Receber como entrada uma imagem contendo o desenho de arquitetura de uma aplicação;
- Processar essa imagem utilizando técnicas de prompt engineering;
- Gerar automaticamente uma análise de ameaças baseada na metodologia STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)

Mas como a vida dá muitas voltas e não tenho mais uma assinatura Azure com créditos de demonstração para obter uma chave API de algum modelo sofisticado, decidi tentar uma alternativa grátis, como a API da Huggingface e também colocar o programa para rodar localmente, através do módulo transformer. O processo em si foi bastante trabalhoso e precisei de uma ajuda ativa bastante significativa do CoPilot, tanto para adicionar linhas de código que fizessem o programa rodar localmente, assim como para debugging, que foi extremamente extensivo. Dessa maneira, se o usuário tiver uma chave Azure, ele pode usá-la, assim como um token Huggingface, ou, se tiver instalada a biblioteca transformer, pode rodar localmente o programa. Nesse último caso, foi baixado localmente o modelo GPT-2, que é antigo, mas serve como a prova de conceito que eu desejava. Dessa maneira, creio que essa nova funcionalidade adiciona muito ao programa.

Abaixo está a documentação completa do programa, gerada pelo próprio CoPilot e revisada, corrigida e expandida por mim.

## STRIDE Demo (local-ready)

Este repositório contém um backend em FastAPI para gerar modelos de ameaças STRIDE a partir de
uma descrição e uma imagem de arquitetura. O front-end está em `module-1/02-front-end/index.html`.

Principais recursos:
- Rota `/analisar_ameacas` aceita upload de imagem e campos de formulário e retorna JSON
- Suporta três backends: `azure` (Azure OpenAI), `hf` (Hugging Face Inference API) e `local` (transformers)
- Quando o modelo não retornar JSON estrito, aplica uma heurística para converter texto livre em JSON

URL do front-end (após iniciar o servidor):

    http://127.0.0.1:8001/

Pré-requisitos
--------------
- Python 3.10+
- Recomendado: criar um virtualenv
- Instalar dependências (exemplo mínimo):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn python-dotenv requests transformers torch
```

Observações:
- Se você pretende usar `BACKEND=local`, instale `transformers` e `torch`. Conexão com GPU acelera significativamente.
- Para usar Hugging Face Inference API (`BACKEND=hf`) defina `HUGGINGFACE_API_TOKEN` no arquivo `.env`.
- Para usar Azure OpenAI (`BACKEND=azure`) defina as variáveis `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` e `AZURE_OPENAI_DEPLOYMENT_NAME`.

Como rodar
----------
1. Copie o arquivo de exemplo de variáveis de ambiente (se existir) ou crie `.env` no diretório `module-1/01-introducao-backend/` com conteúdo como:

```
BACKEND=local
HUGGINGFACE_MODEL=distilgpt2
# HUGGINGFACE_API_TOKEN=...
# AZURE_OPENAI_API_KEY=...
```

2. Inicie o servidor (sem `--reload` para evitar conflitos de reloader):

```powershell
cd module-1/01-introducao-backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

3. Abra a URL `http://127.0.0.1:8001/` no navegador para acessar o front-end. Envie uma imagem PNG e preencha os campos. **Caso deseje subir uma imagem jpg/jpeg, ou qualquer outro formato, isso deve ser modificado no programa.**

Testes locais
-------------
- Há scripts de teste no diretório para chamadas in-process (TestClient) e testes HTTP; veja `http_test_inprocess.py` e `http_test_post.py`.

Suporte e melhorias
-------------------
- A heurística de conversão de texto para JSON é simples; para resultados mais confiáveis use um modelo maior
  e/ou inclua exemplos (few-shot) no prompt.
- Se quiser, posso melhorar a heurística ou adicionar exemplos no prompt (few-shot) para aumentar a probabilidade de saída JSON.

---
Arquivo principal do backend: `module-1/01-introducao-backend/main.py`

# STRIDE Threat Model Analyzer

Este projeto é uma solução completa para análise de ameaças baseada na metodologia STRIDE, composta por um backend em FastAPI (Python) e um front-end em HTML/CSS/JS com visualização de ameaças usando Cytoscape.js.

## Funcionalidades
- Upload de imagem de arquitetura e preenchimento de informações do sistema.
- Geração automática de modelo de ameaças STRIDE usando Azure OpenAI.
- Visualização do modelo de ameaças em grafo interativo (Cytoscape.js).
- Sugestões de melhoria para o modelo de ameaças.
- Botão para imprimir/exportar o grafo gerado.

---

## Como executar o projeto

### 1. Pré-requisitos
- Python 3.10+
- Node.js (opcional, apenas se quiser servir o front-end com algum servidor local)
- Conta e deployment configurado no Azure OpenAI (veja variáveis de ambiente)

### 2. Clonando o repositório

```bash
# Clone o projeto
 git clone https://github.com/digitalinnovationone/stride-demo.git
 cd stride-demo
```

### 3. Configurando o backend (FastAPI)

1. Acesse a pasta do backend:
   ```bash
   cd module-1/01-introducao-backend
   ```
2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Crie um arquivo `.env` com as seguintes variáveis (preencha com seus dados do Azure OpenAI):
   ```env
   AZURE_OPENAI_API_KEY=xxxxxx # nesse caso, o usuário pode escolher múltiplos modelos que estão na vanguarda da tecnologia, como o o1, o3, GPT 4o, GPT 5 etc... 
   AZURE_OPENAI_ENDPOINT=https://<seu-endpoint>.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2023-05-15
   AZURE_OPENAI_DEPLOYMENT_NAME=<nome-do-deployment>
   HUGGINGFACE_API_TOKEN=<seu token Hugginface>
   HUGGINGFACE_MODEL=distilgpt2 #uma versão grátis dessa API roda no GPT2
   BACKEND=local #rodando localmente, também no GPT2
   ```
5. Execute o backend:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```
   O backend estará disponível em http://localhost:8001

### 4. Configurando o front-end

1. Acesse a pasta do front-end:
   ```bash
   cd ../02-front-end
   ```
2. Basta abrir o arquivo `index.html` no navegador (duplo clique ou `open index.html`).
   - Se quiser servir via servidor local (opcional):
     ```bash
     npx serve .
     # ou
     python -m http.server 8080
     ```
3. O front-end espera que o backend esteja rodando em http://localhost:8001, via univorn, como descrito acima.

---

### Cuidados e dicas
- **Azure OpenAI:** Certifique-se de que seu deployment está ativo e as variáveis do `.env` estão corretas.
- **CORS:** O backend já está configurado para aceitar requisições de qualquer origem, mas se for usar em produção, ajuste as origens permitidas.
- **Limite de tokens:** O modelo do Azure OpenAI pode ter limites de tokens. Ajuste `max_tokens` se necessário.
- **Impressão do grafo:** O botão "Imprimir Grafo" exporta a visualização atual do grafo como imagem para impressão ou PDF.
- **Formato do JSON:** O front-end espera o JSON no formato retornado pelo backend. Se mudar o backend, ajuste o front-end conforme necessário.
- **Portas:** Certifique-se de que as portas 8001 (backend) e 8080 (front-end, se usar servidor) estejam livres.

---

## Estrutura do projeto
```
stride-demo/
│
├── module-1/
│   ├── 01-introducao-backend/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── .env (crie este arquivo)
│   └── 02-front-end/
│       └── index.html
└── README.md
```

---

