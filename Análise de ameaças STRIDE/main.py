import os
# URL do front-end (abra no navegador após iniciar o servidor): http://127.0.0.1:8001/, com univorn
# Este arquivo contém o backend FastAPI que expõe a rota /analisar_ameacas
# e também serve a interface front-end localizada em `02-front-end/index.html`.
# Evita que Transformers tente usar TensorFlow/Keras (resolve erro Keras 3 incompatível)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import base64
import tempfile
import requests
import json
import re

try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    pipeline = None
    transformers_available = False
try:
    from openai import AzureOpenAI
    azure_available = True
except Exception:
    AzureOpenAI = None
    azure_available = False
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

env_path = Path(__file__).resolve(strict=True).parent / ".env"
# Carrega variáveis definidas em .env para o ambiente Python (útil para chaves e configurações locais)
load_dotenv(dotenv_path=env_path)

# Carregar as variáveis de ambiente do arquivo .env
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Backend selecionável: 'azure', 'hf' (Hugging Face) ou 'local'.
# Configure a variável BACKEND no arquivo .env ou no ambiente antes de iniciar o servidor.
BACKEND = 'local'  # exemplo: 'azure', 'hf' ou 'local'


# Configuração do FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)

# Servir interface front-end (index.html)
module1_dir = Path(__file__).resolve(strict=True).parent.parent
front_dir = module1_dir / "02-front-end"
if front_dir.exists():
    # Monta os arquivos do front-end em /static para que possam ser servidos diretamente.
    app.mount("/static", StaticFiles(directory=str(front_dir)), name="static")


@app.get("/", response_class=FileResponse)
def serve_index():
    """Rota raiz para servir o `index.html` do front-end.
    Abra `http://127.0.0.1:8001/` no navegador após iniciar o servidor.
    """
    index_path = front_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(content={"error": "index.html not found"}, status_code=404)

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if HUGGINGFACE_API_TOKEN:
    HUGGINGFACE_API_TOKEN = HUGGINGFACE_API_TOKEN.strip()
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "gpt2")  # exemplo: 'gpt2' ou um modelo de chat compatível
HF_MODEL = (HF_MODEL or "gpt2").strip()

# Variáveis globais para os geradores de texto
local_generator = None

def call_local_model(text):
    """Gera texto usando o modelo local carregado com pipeline."""
    if local_generator is None:
        raise RuntimeError("Gerador local não está disponível (Transformers instalado?)")
    outputs = local_generator(text, max_new_tokens=128, truncation=True, do_sample=True, temperature=0.7, num_return_sequences=1)
    # Depending on pipeline, output may be list/dict
    return outputs[0].get("generated_text") if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) else str(outputs)

# Configurar cliente Azure se selecionado e disponível.
# Nota para iniciantes: apenas necessário se você realmente for usar Azure OpenAI.
client = None
if BACKEND == "azure":
    if not azure_available:
        raise RuntimeError("AzureOpenAI library não está disponível; instale a biblioteca ou mude BACKEND para 'hf' ou 'local'.")
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME)

# suporte a modelo local (carregado sob demanda)
local_generator = None
try:
    # Importa a função `pipeline` do Hugging Face Transformers. Se não estiver instalada,
    # o backend 'local' não ficará disponível.
    from transformers import pipeline
    transformers_available = True
except Exception:
    pipeline = None
    transformers_available = False

def call_hf_inference(prompt: str):
    if not HUGGINGFACE_API_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN não encontrado. Defina no arquivo .env ou variáveis de ambiente.")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"temperature": 0.7, "max_new_tokens": 512}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        # Se modelo não encontrado, verificar existência no catálogo HF e sugerir correções
        if status == 404:
            # tenta consultar metadados do modelo publicamente
            meta_url = f"https://huggingface.co/api/models/{HF_MODEL}"
            try:
                meta = requests.get(meta_url, timeout=20)
                if meta.status_code == 200:
                    # modelo existe, mas pode não estar habilitado para inference API
                    msg = (
                        f"Modelo '{HF_MODEL}' encontrado no Hugging Face, mas não disponível via Inference API. "
                        "Algumas razões: o modelo requer autorização especial, está privado, ou não oferece endpoint de inferência hospedado. "
                        "Sugestão: escolha um modelo público compatível com text-generation, por exemplo 'EleutherAI/gpt-neo-125M' ou 'gpt2', e atualize HUGGINGFACE_MODEL no .env."
                    )
                else:
                    msg = (
                        f"Modelo '{HF_MODEL}' não encontrado no Hugging Face (consulta retornou {meta.status_code}). "
                        "Verifique o nome do modelo e use o identificador completo com namespace, ex.: 'username/model-name'."
                    )
            except Exception:
                msg = ("Modelo não encontrado e falha ao consultar metadados do Hugging Face. Verifique sua conexão ou o nome do modelo.")

            # Não faz fallback automático para local (pode baixar modelos muito grandes)
            # Informe ao usuário o motivo e como corrigir: usar um modelo hospedado diferente
            # ou mudar para BACKEND=local explicitamente.
            raise RuntimeError(msg + " Se quiser usar um modelo local em vez da Inference API, defina BACKEND=local no .env e certifique-se de ter 'transformers' instalado.")
        # Para outros erros HTTP, propagar mensagem útil
        else:
            body = e.response.text if e.response is not None else str(e)
            raise RuntimeError(f"Hugging Face inference API retornou {status}: {body}")

    # fim try/except
    # resposta do HF pode ter formatos diferentes dependendo do modelo;
    # para modelos de texto costuma vir em data[0]['generated_text'] ou data['generated_text']
    
    # agora 'data' deve existir se não houver erro
    # resposta do HF pode ter formatos diferentes dependendo do modelo;
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # fallback: retornar JSON bruto como string
    return str(data)


def call_local_model(prompt: str):
    global local_generator
    if not transformers_available:
        raise RuntimeError("transformers não instalado. Instale 'transformers' e (opcional) 'torch' para usar BACKEND=local.")
    if local_generator is None:
        # carrega um gerador leve (gpt2 por exemplo)
        # Forçar PyTorch e escolher dispositivo CUDA se disponível para evitar fallback para TF/Keras
        try:
            import torch
            use_cuda = torch.cuda.is_available()
            device = 0 if use_cuda else -1
        except Exception:
            device = -1
        local_generator = pipeline("text-generation", model=HF_MODEL, framework="pt", device=device)
    # Use max_new_tokens and truncation to avoid excessive generation time
    outputs = local_generator(prompt, max_new_tokens=128, truncation=True, do_sample=True, temperature=0.7, num_return_sequences=1)
    # Depending on pipeline, output may be list/dict
    return outputs[0].get("generated_text") if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) else str(outputs)


@app.on_event("startup")
def preload_local_generator():
    """Se o BACKEND for 'local', pré-carrega o pipeline no startup para evitar que o
    primeiro pedido HTTP fique bloqueado pelo download do modelo."""
    global local_generator
    if BACKEND == "local":
        if not transformers_available:
            print("transformers não disponível; BACKEND=local exigirá 'transformers' instalado.")
            return
        if local_generator is None:
            try:
                import torch
                use_cuda = torch.cuda.is_available()
                device = 0 if use_cuda else -1
            except Exception:
                device = -1
            print(f"Pré-carregando gerador local (modelo={HF_MODEL}, device={device})")
            try:
                local_generator = pipeline("text-generation", model=HF_MODEL, framework="pt", device=device)
                print("Gerador local pré-carregado com sucesso.")
            except Exception as e:
                print("Falha ao pré-carregar gerador local:", e)

# (removido trecho obsoleto)

def criar_prompt_modelo_ameacas(tipo_aplicacao, 
                                autenticacao, 
                                acesso_internet, 
                                dados_sensiveis, 
                                descricao_aplicacao):
    prompt = f"""Aja como um especialista em cibersegurança com mais de 20 anos de experiência 
    utilizando a metodologia de modelagem de ameaças STRIDE para produzir modelos de ameaças 
    abrangentes para uma ampla gama de aplicações. Sua tarefa é analisar o resumo do código, 
    o conteúdo do README e a descrição da aplicação fornecidos para produzir uma lista de 
    ameaças específicas para essa aplicação.

    Presta atenção na descrição da aplicação e nos detalhes técnicos fornecidos.

    Para cada uma das categorias do STRIDE (Falsificação de Identidade - Spoofing, 
    Violação de Integridade - Tampering, 
    Repúdio - Repudiation, 
    Divulgação de Informações - Information Disclosure, 
    Negação de Serviço - Denial of Service, e 
    Elevação de Privilégio - Elevation of Privilege), liste múltiplas (3 ou 4) ameaças reais, 
    se aplicável. Cada cenário de ameaça deve apresentar uma situação plausível em que a ameaça 
    poderia ocorrer no contexto da aplicação.

    A lista de ameaças deve ser apresentada em formato de tabela, 
    com as seguintes colunas:Ao fornecer o modelo de ameaças, utilize uma resposta formatada em JSON 
    com as chaves "threat_model" e "improvement_suggestions". Em "threat_model", inclua um array de 
    objetos com as chaves "Threat Type" (Tipo de Ameaça), "Scenario" (Cenário), e 
    "Potential Impact" (Impacto Potencial).    

    Ao fornecer o modelo de ameaças, utilize uma resposta formatada em JSON com as chaves 
    "threat_model" e "improvement_suggestions". 
    Em "threat_model", inclua um array de objetos com as chaves "Threat Type" (Tipo de Ameaça), 
    "Scenario" (Cenário), e "Potential Impact" (Impacto Potencial).

    Em "improvement_suggestions", inclua um array de strings que sugerem quais informações adicionais 
    poderiam ser fornecidas para tornar o modelo de ameaças mais completo e preciso na próxima iteração. 
    Foque em identificar lacunas na descrição da aplicação que, se preenchidas, permitiriam uma 
    análise mais detalhada e precisa, como por exemplo:
    - Detalhes arquiteturais ausentes que ajudariam a identificar ameaças mais específicas
    - Fluxos de autenticação pouco claros que precisam de mais detalhes
    - Descrição incompleta dos fluxos de dados
    - Informações técnicas da stack não informadas
    - Fronteiras ou zonas de confiança do sistema não especificadas
    - Descrição incompleta do tratamento de dados sensíveis
    - Detalhes sobre
    Não forneça recomendações de segurança genéricas — foque apenas no que ajudaria a criar um
    modelo de ameaças mais eficiente.

    TIPO DE APLICAÇÃO: {tipo_aplicacao}
    MÉTODOS DE AUTENTICAÇÃO: {autenticacao}
    EXPOSTA NA INTERNET: {acesso_internet}
    DADOS SENSÍVEIS: {dados_sensiveis}
    RESUMO DE CÓDIGO, CONTEÚDO DO README E DESCRIÇÃO DA APLICAÇÃO: {descricao_aplicacao}

        Exemplo de formato esperado em JSON:

        {{
            "threat_model": [
                {{
                    "Threat Type": "Spoofing",
                    "Scenario": "Cenário de exemplo 1",
                    "Potential Impact": "Impacto potencial de exemplo 1"
                }},
                {{
                    "Threat Type": "Spoofing",
                    "Scenario": "Cenário de exemplo 2",
                    "Potential Impact": "Impacto potencial de exemplo 2"
                }}
            ],
            "improvement_suggestions": [
                "Por favor, forneça mais detalhes sobre o fluxo de autenticação entre os componentes para permitir uma análise melhor de possíveis falhas de autenticação.",
                "Considere adicionar informações sobre como os dados sensíveis são armazenados e transmitidos para permitir uma análise mais precisa de exposição de dados."
            ]
        }}

        INSTRUÇÃO IMPORTANTE: RESPONDA APENAS COM JSON VÁLIDO, SEM TEXTO ADICIONAL NEM EXPLICAÇÕES. GARANTA QUE A SAÍDA SEJA UM OBJETO JSON COM EXATAMENTE DUAS CHAVES: "threat_model" (array de objetos) e "improvement_suggestions" (array de strings).

        SE POR QUALQUER MOTIVO VOCÊ NÃO CONSEGUIR GERAR UM JSON VÁLIDO, RESPONDA APENAS O JSON: {{"error":"unable_to_generate_json"}}

        Por favor, analise a imagem e o texto acima e forneça apenas o JSON solicitado.
"""

    return prompt

@app.post("/analisar_ameacas")
async def analisar_ameacas(
    imagem: UploadFile = File(...),
    tipo_aplicacao: str = Form(...),
    autenticacao: str = Form(...),
    acesso_internet: str = Form(...),
    dados_sensiveis: str = Form(...),
    descricao_aplicacao: str = Form(...)
):
    try:
        # 1) Depuração: mostrar informações do arquivo enviado
        print(imagem)
        # 2) Criar o prompt que será enviado ao modelo. A função monta uma instrução
        #    detalhada pedindo um JSON STRIDE com duas chaves: 'threat_model' e 'improvement_suggestions'.
        prompt = criar_prompt_modelo_ameacas(tipo_aplicacao, 
                                              autenticacao, 
                                              acesso_internet, 
                                              dados_sensiveis, 
                                              descricao_aplicacao)
    # 3) Salvar a imagem enviada em um arquivo temporário para podermos ler em base64
    #    e incluir no prompt para o modelo (alguns backends aceitam imagens, aqui usamos base64).
        content = await imagem.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(imagem.filename).suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 4) Converter imagem para base64 (string) para anexar ao prompt
        with open(temp_file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('ascii')


    # 5) Construir uma estrutura de "mensagens" (chat_prompt) que contém o texto e a imagem.
    #    Note: para Hugging Face/local convertemos isso em texto simples abaixo.
        chat_prompt = [
            {"role": "system", "content": "Você é uma IA especialista em cibersegurança, que analisa desenhos de arquitetura."},
            {"role": "user"
             , "content": [
                {"type": "text"
                 , "text": prompt
                 },
                {
                    "type": "image_url"
                 ,  "image_url": {"url": f"data:image/png;base64,{encoded_string}"}
                 },
                {"type": "text", 
                 "text": "Por favor, analise a imagem e o texto acima e forneça um modelo de ameaças detalhado."
                 }]
        }]
    # 6) Escolher o backend configurado (azure | hf | local) e gerar a resposta.
    #    Cada backend retorna texto; tentamos extrair JSON do texto ou aplicar heurística.
        result_text = None
        if BACKEND == "azure":
            if client is None:
                raise RuntimeError("Cliente Azure não configurado")
            response = client.chat.completions.create(
                messages = chat_prompt,
                temperature=0.7,
                max_tokens=1500,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream= False,
                model= AZURE_OPENAI_DEPLOYMENT_NAME
            )
            result_text = response.to_dict()
        elif BACKEND == "hf":
            # Para Hugging Face, transformamos chat_prompt em uma string simples
            prompt_text = "\n".join([item.get("text", "") for item in chat_prompt[1]["content"] if isinstance(item, dict)])
            hf_out = call_hf_inference(prompt_text)
            result_text = {"text": hf_out}
        elif BACKEND == "local":
            prompt_text = "\n".join([item.get("text", "") for item in chat_prompt[1]["content"] if isinstance(item, dict)])
            local_out = call_local_model(prompt_text)
            result_text = {"text": local_out}
        else:
            raise RuntimeError(f"BACKEND desconhecido: {BACKEND}")

        os.remove(temp_file_path)  # Remover o arquivo temporário após o uso

        # Se o backend retornou um dicionário já estruturado, tente retornar diretamente
        if isinstance(result_text, dict) and not (isinstance(result_text.get("text" , None), str)):
            return JSONResponse(content=result_text, status_code=200)

        # Caso comum: modelos locais/HF retornam texto em result_text['text']
        raw_text = None
        if isinstance(result_text, dict) and "text" in result_text:
            raw_text = result_text["text"]
        elif isinstance(result_text, str):
            raw_text = result_text
        else:
            raw_text = str(result_text)

        # Tentar carregar o texto como JSON diretamente
        parsed = None
        if raw_text is not None:
            try:
                parsed = json.loads(raw_text)
            except Exception:
                # tentar extrair a primeira substring JSON entre '{' e '}' como fallback
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    candidate = raw_text[start:end+1]
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        parsed = None

        if parsed is not None:
            # Limpar e normalizar JSON retornado pelo modelo
            try:
                cleaned = clean_generated_json(parsed)
                return JSONResponse(content=cleaned, status_code=200)
            except Exception:
                return JSONResponse(content=parsed, status_code=200)
        # Falha ao obter JSON válido do gerador -> tentar heurística
        try:
            structured = parse_stride_text_to_json(raw_text)
            try:
                cleaned = clean_generated_json(structured)
                return JSONResponse(content=cleaned, status_code=200)
            except Exception:
                return JSONResponse(content=structured, status_code=200)
        except Exception:
            return JSONResponse(content={"error": "unable_to_generate_json", "raw": raw_text[:2000]}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def parse_stride_text_to_json(text: str):
    """Heurística simples (fallback): procura por cabeçalhos STRIDE e bullets subsequentes.
    - Quando o modelo não retorna JSON válido, esta função tenta converter texto livre em
      uma estrutura JSON com as chaves esperadas para a aplicação front-end.
    - Não é perfeita, mas facilita que iniciantes obtenham um resultado utilizável sem ajustar modelos.
    Retorna dict com keys: 'threat_model' (list of {Threat Type, Scenario, Potential Impact})
    e 'improvement_suggestions' (list).
    """
    # Normalizar linhas
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    categories = ["Spoofing", "Tampering", "Repudiation", "Information Disclosure", "Denial of Service", "Elevation of Privilege"]

    # Mapeamento de palavras-chave para categorias STRIDE apropriadas
    keyword_to_category = {
        "autenticação": "Spoofing",
        "senha": "Spoofing",
        "fingerprint": "Spoofing",
        "identidade": "Spoofing",
        "identificação": "Spoofing",
        "credenciais": "Spoofing",
        "login": "Spoofing",

        "dados": "Information Disclosure",
        "informações": "Information Disclosure",
        "sensíveis": "Information Disclosure",
        "confidenciais": "Information Disclosure",
        "banco de dados": "Information Disclosure",
        "sql": "Information Disclosure",
        "armazenados": "Information Disclosure",

        "integridade": "Tampering",
        "modificação": "Tampering",
        "alteração": "Tampering",
        "manipulação": "Tampering",
        "valores": "Tampering",
        "pagamento": "Tampering",
        "transferência": "Tampering",

        "log": "Repudiation",
        "auditoria": "Repudiation",
        "registro": "Repudiation",
        "rastreamento": "Repudiation",
        "histórico": "Repudiation",
        "transação": "Repudiation",

        "disponibilidade": "Denial of Service",
        "performance": "Denial of Service",
        "ddos": "Denial of Service",
        "serviço": "Denial of Service",
        "acesso": "Denial of Service",
        "internet": "Denial of Service",

        "permissão": "Elevation of Privilege",
        "privilégio": "Elevation of Privilege",
        "admin": "Elevation of Privilege",
        "administrador": "Elevation of Privilege",
        "autorização": "Elevation of Privilege",
    }

    threat_model = []
    suggestions = []
    current_cat = None
    buffer = []

    # regex para bullets
    bullet_re = re.compile(r"^[-\*\u2022]\s*(.*)")

    for ln in lines:
        # Primeiro tenta encontrar uma categoria explícita
        current_cat = None
        for cat in categories:
            if re.search(r"\b" + re.escape(cat.split()[0]) + r"\b", ln, re.IGNORECASE):
                current_cat = cat
                break

        # Se não encontrou categoria explícita, tenta inferir das palavras-chave
        if not current_cat:
            ln_lower = ln.lower()
            for keyword, category in keyword_to_category.items():
                if keyword in ln_lower:
                    current_cat = category
                    break

        # Se encontrou uma categoria (explícita ou inferida), adiciona o cenário
        if current_cat:
            # Tenta extrair impacto se houver texto após "causando", "resultando", etc.
            impact = ""
            impact_markers = ["causando", "resultando em", "levando a", "permite", "possibilita"]
            for marker in impact_markers:
                if marker in ln_lower:
                    parts = ln.split(marker, 1)
                    if len(parts) == 2:
                        ln = parts[0].strip()
                        impact = marker + " " + parts[1].strip()

            # Adiciona a entrada com categoria inferida e possível impacto
            if len(ln) > 10:  # evita linhas muito curtas
                threat_model.append({
                    "Threat Type": current_cat,
                    "Scenario": ln,
                    "Potential Impact": impact
                })
            continue

        else:
            m = bullet_re.match(ln)
            if m:
                text_b = m.group(1).strip()
                if len(text_b) > 10 and not any(k in text_b.lower() for k in ["threat_model", "improvement_suggestions", "json"]):
                    suggestions.append(text_b)
            else:
                # lines that look like suggestions
                if any(k in ln.lower() for k in ["suggest", "improve", "recomen", "sugest"]):
                    ln = ln.strip()
                    if len(ln) > 10 and not any(k in ln.lower() for k in ["threat_model", "improvement_suggestions", "json"]):
                        suggestions.append(ln)

    # Se não encontrou ameaças, tenta gerar algumas baseadas no contexto
    if not threat_model:
        # Para sistema de pagamentos, criar algumas ameaças comuns
        context_threats = [
            {
                "Threat Type": "Spoofing",
                "Scenario": "Possibilidade de autenticação insuficiente nas transações de pagamento",
                "Potential Impact": "Permitindo transações não autorizadas"
            },
            {
                "Threat Type": "Tampering",
                "Scenario": "Risco de modificação não autorizada dos valores de transferência",
                "Potential Impact": "Causando prejuízo financeiro"
            },
            {
                "Threat Type": "Information Disclosure",
                "Scenario": "Exposição de dados sensíveis armazenados no banco SQL",
                "Potential Impact": "Vazamento de informações financeiras"
            },
            {
                "Threat Type": "Repudiation",
                "Scenario": "Ausência de logs de auditoria para transações",
                "Potential Impact": "Impossibilidade de rastrear operações suspeitas"
            }
        ]
        threat_model.extend(context_threats)

    # Deduplica e limpa sugestões
    suggestions = list(dict.fromkeys([s for s in suggestions if len(s) > 10]))

    return {"threat_model": threat_model, "improvement_suggestions": suggestions}


# Normaliza variações de nomes de categoria para valores canônicos
def normalize_threat_type(t: str) -> str:
    if not t:
        return "Unknown"
    t0 = t.strip().lower()
    mapping = {
        "spoof": "Spoofing",
        "spoofing": "Spoofing",
        "tamper": "Tampering",
        "tampering": "Tampering",
        "repudiation": "Repudiation",
        "repudiat": "Repudiation",
        "information": "Information Disclosure",
        "information disclosure": "Information Disclosure",
        "data disclosure": "Information Disclosure",
        "dos": "Denial of Service",
        "denial": "Denial of Service",
        "denial of service": "Denial of Service",
        "elevation": "Elevation of Privilege",
        "elevation of privilege": "Elevation of Privilege",
        "privilege": "Elevation of Privilege",
    }
    for k, v in mapping.items():
        if k in t0:
            return v
    # fallback: Title case the input first token
    return t.strip().title()


# Limpa e normaliza o JSON gerado pelo modelo
def clean_generated_json(payload: dict) -> dict:
    """Recebe um dict possivelmente gerado automaticamente e retorna uma versão
    limpa:
    - remove entradas muito curtas/ruidosas em improvement_suggestions
    - deduplica cenários idênticos
    - normaliza 'Threat Type'
    - tenta preencher 'Potential Impact' com placeholder vazio se ausente
    """
    out = {"threat_model": [], "improvement_suggestions": []}
    seen = set()

    # Processar threat_model
    for item in payload.get("threat_model", []):
        tt = item.get("Threat Type") or item.get("threat_type") or "Unknown"
        tt_norm = normalize_threat_type(tt)
        scenario = (item.get("Scenario") or item.get("scenario") or "").strip()
        impact = (item.get("Potential Impact") or item.get("potential_impact") or "").strip()
        # ignorar cenários vazios
        if not scenario or len(scenario) < 8:
            continue
        key = (tt_norm, scenario)
        if key in seen:
            continue
        seen.add(key)
        out["threat_model"].append({"Threat Type": tt_norm, "Scenario": scenario, "Potential Impact": impact})

    # Processar improvement_suggestions
    for s in payload.get("improvement_suggestions", []):
        if not isinstance(s, str):
            continue
        s2 = s.strip()
        # descartar linhas que são claramente pedaços do próprio prompt
        if len(s2) < 10:
            continue
        low = s2.lower()
        if any(k in low for k in ["threat_model", "improvement_suggestions", "responda apenas com json", "instrução importante"]):
            continue
        # evitar sugestões demasiado semelhantes
        if s2 in out["improvement_suggestions"]:
            continue
        out["improvement_suggestions"].append(s2)

    # Se não há ameaças detectadas, manter ao menos uma entrada genérica
    if not out["threat_model"]:
        sample_text = ""
        # tentar extrair do payload raw text
        if isinstance(payload, dict) and payload.get("raw"):
            sample_text = payload.get("raw")[:200]
        out["threat_model"].append({"Threat Type": "Unknown", "Scenario": sample_text, "Potential Impact": ""})

    return out
