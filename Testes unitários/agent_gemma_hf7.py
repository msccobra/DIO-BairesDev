# agent_gemma_hf.py
import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
import torch
from string import Template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import pytest
import io, contextlib, threading, queue, time

# Carrega variáveis do .env
env_path = r'C:\Users\flawl\Documentos\Programas\Testes DIO\.env'
load_dotenv(dotenv_path=env_path)

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if not HF_TOKEN:
    raise RuntimeError("Defina HUGGINGFACE_HUB_TOKEN no ambiente ou faça hf auth login.")

# Login para modelos gated
login(token=HF_TOKEN)

# Configurações do modelo Gemma
MODEL_NAME = os.getenv("GEMMA_MODEL", "google/gemma-2-2b")
MAX_NEW_TOKENS = int(os.getenv("GEMMA_MAX_NEW_TOKENS","512"))  # reduzido para evitar excesso
TEMPERATURE = float(os.getenv("GEMMA_TEMPERATURE", "0.8"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)

if torch.cuda.is_available():
    # half-precision economiza VRAM e aumenta velocidade
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        dtype=torch.float16,   # fp16 na GPU
        device_map="auto",           # deixa o HF/accelerate dividir entre GPUs se houver mais de uma
        trust_remote_code=True
    )
    # `device=0` força o pipeline a enviar tensores para a GPU 0
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device=0,                   # GPU
        trust_remote_code=True
    )
else:
    # CPU: usa dtype default (fp32)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    ).to("cpu")

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,                  # CPU
        trust_remote_code=True
    )

def run_pytest_and_parse(folder: str, timeout: int = 20) -> Dict:
    """
    Executa pytest dentro do mesmo processo, captura a saída e devolve
    um dicionário de resumo semelhante ao usado anteriormente.
    """

    def _worker(q: "queue.Queue[str]"):
        buff_out = io.StringIO()
        buff_err = io.StringIO()
        with contextlib.redirect_stdout(buff_out), contextlib.redirect_stderr(buff_err):
            # -q = quiet, --disable-warnings para silenciar barulho extra
            rc = pytest.main(
        ["-q", "--disable-warnings", "test_generated.py"],
        plugins=[]
    )
        q.put((rc, buff_out.getvalue() + "\n" + buff_err.getvalue()))

    q: "queue.Queue[tuple[int,str]]" = queue.Queue()
    t = threading.Thread(target=_worker, args=(q,), daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return {
            "returncode": -1,
            "summary_line": "Timeout ao rodar pytest",
            "success": False,
        }

    returncode, output = q.get()

    # --- obtém a última linha contendo passed/failed/errors -----------
    summary_line = ""
    for line in output.splitlines()[::-1]:
        if line.strip().startswith("===") or re.search(r"passed|failed|errors", line):
            summary_line = line.strip()
            break

    result = {
        "returncode": returncode,
        "summary_line": summary_line,
        "full_output": output,
        "success": returncode == 0,
    }

    m = re.search(
        r"(?:(\d+)\s+passed)?(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+error)?",
        summary_line,
    )
    if m:
        result.update(
            {
                "passed": int(m.group(1) or 0),
                "failed": int(m.group(2) or 0),
                "errors": int(m.group(3) or 0),
            }
        )
    return result

# Template do prompt

PROMPT_TEMPLATE = Template(
"""Crie TESTES pytest para o módulo $module.

REGRAS
• Não repita o código do módulo.
• Todos os nomes de funções DEVEM começar com `test_`.
• Use apenas asserts simples (sem classes, sem fixtures).
• Devolva APENAS um bloco ```python```.

EXEMPLO
```python
import pytest
from math import sqrt

def test_sqrt():
    assert sqrt(4) == 2
"""
)

def read_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

import textwrap, itertools, ast, re

import textwrap, ast, re

# ----------------------------------------------------------------------
# substitua call_gemma_generate() pela versão abaixo
# ----------------------------------------------------------------------
import ast, re, textwrap

_BLOCK_RE = re.compile(r"```(?:python)?(.*?)```", re.S | re.I)

def _sanitize_code(raw: str) -> str:
    """
    Extrai apenas o código Python da resposta do modelo.

    1) Se houver blocos ``` ``` pega o 1º bloco.
    2) Caso contrário, mantém a partir da 1ª linha que começa com
       'import', 'from' ou 'def test'.
    3) Remove linhas com acentuação/PT-BR e garante que compila.
    """
    m = _BLOCK_RE.search(raw)
    code = m.group(1) if m else raw

    # começa onde surge código de teste
    for i, line in enumerate(code.splitlines()):
        if line.lstrip().startswith(("import", "from", "def test")):
            code = "\n".join(code.splitlines()[i:])
            break

    # remove linhas em português / acentuadas
    clean = [
        ln for ln in code.splitlines()
        if not re.search(r"[áéíóúâêôãõç]", ln, re.I)
    ]
    code = "\n".join(clean)

    # tenta dedent + validar sintaxe
    code = textwrap.dedent(code).strip()
    try:
        ast.parse(code)
    except SyntaxError:
        # se falhar, reduz até a 1ª 'def test'
        m = re.search(r"^\s*def\s+test", code, re.M)
        if m:
            code = code[m.start():]
    return code.strip()


import ast, re, textwrap

def build_local_tests(module_name: str, path: str) -> str:
    """
    Caso o modelo não gere testes, cria um esqueleto local cobrindo
    todas as funções públicas do módulo.
    """
    import importlib.util, inspect, textwrap, sys, types, pathlib

    spec = importlib.util.spec_from_file_location(module_name, path)
    mod: types.ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    funcs = [
        f for f in vars(mod).values()
        if inspect.isfunction(f) and f.__module__ == module_name
           and not f.__name__.startswith('_')
    ]

    code = ["import pytest", f"import {module_name} as m", ""]
    for f in funcs:
        code.append(f"def test_{f.__name__}():")
        sig = inspect.signature(f)
        params = ", ".join(
            "0" if p.default is inspect._empty else repr(p.default)
            for p in sig.parameters.values()
        ) or ""
        call = f"m.{f.__name__}({params})"
        code.append(f"    _ = {call}  # TODO: ajustar asserts")
        code.append("")
    return "\n".join(code)

def call_gemma_generate(prompt: str,
                        module_name: str,
                        module_path: str,
                        max_attempts: int = 3) -> str:
    for _ in range(max_attempts):
        raw = gen_pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )[0]["generated_text"]

        # sanitiza (igual você já tinha)
        code = _sanitize_code(raw)

        # aceita se compila E tem pelo menos 1 def test_
        try:
            ast.parse(code)
            if re.search(r"\bdef\s+test_\w+", code):
                return code
        except SyntaxError:
            pass
        # reforça o prompt na próxima volta
        prompt += "\n# Gere somente código válido com def test_*\n"

    # --- fallback local -------------------------------------------------
    print("\n🔄  Modelo falhou 3× – gerando esqueleto local.")
    return build_local_tests(module_name, module_path)

def write_test_file(content: str, source_path: str) -> str:
    p = Path(source_path).with_name("test_generated.py")
    if not content.startswith("import"):
        content = "import pytest\n\n" + content
    p.write_text(content, encoding="utf-8")
    return str(p)


def run_pytest_and_parse(folder: str, timeout: int = 20) -> Dict:
    try:
        cmd = ["pytest", "-q", "--disable-warnings", "--maxfail=0"]
        proc = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "summary_line": "Timeout ao rodar pytest", "success": False}

    summary_line = ""
    for line in out.splitlines()[::-1]:
        if line.strip().startswith("===") or re.search(r"passed|failed|errors", line):
            summary_line = line.strip()
            break

    result = {
        "returncode": proc.returncode,
        "summary_line": summary_line,
        "full_output": out,
        "success": proc.returncode == 0
    }

    m = re.search(r"(?:(\d+)\s+passed)?(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+error)?", summary_line)
    if m:
        result.update({
            "passed": int(m.group(1) or 0),
            "failed": int(m.group(2) or 0),
            "errors": int(m.group(3) or 0),
        })
    return result

print(">>> Versão in-process de run_pytest_and_parse carregada")  # ← sentinela
import io, contextlib, threading, queue, re, pytest, os
from typing import Dict

def run_pytest_and_parse(folder: str, timeout: int = 20) -> Dict:
    """
    Executa pytest no mesmo processo e devolve estatísticas.
    """
    def _worker(q: "queue.Queue"):
        buff = io.StringIO()
        here = os.getcwd()
        try:
            os.chdir(folder)
            with contextlib.redirect_stdout(buff), contextlib.redirect_stderr(buff):
                rc = pytest.main(
        ["-q", "--disable-warnings", "test_generated.py"],
        plugins=[]
    )
        finally:
            os.chdir(here)
        q.put((rc, buff.getvalue()))

    q: "queue.Queue" = queue.Queue()
    t = threading.Thread(target=_worker, args=(q,), daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return {"returncode": -1, "summary_line": "Timeout ao rodar pytest", "success": False}

    returncode, output = q.get()
    summary_line = ""
    for line in output.splitlines()[::-1]:
        if re.search(r"\bpassed\b|\bfailed\b|\berrors?\b", line):
            summary_line = line.strip()
            break

    result = {
        "returncode": returncode,
        "summary_line": summary_line,
        "full_output": output,
        "success": returncode == 0,
    }

    m = re.search(r"(?:(\d+)\s+passed)?(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+errors?)?", summary_line)
    if m:
        result.update({
            "passed": int(m.group(1) or 0),
            "failed": int(m.group(2) or 0),
            "errors": int(m.group(3) or 0),
        })
    return result


def generate_tests_for_file(source_path: str) -> Dict:
    source_code = read_source(source_path)
    module_name = Path(source_path).stem
    prompt = PROMPT_TEMPLATE.safe_substitute(module=module_name)
    tests_code = call_gemma_generate(prompt, module_name, str(Path(source_path).resolve()))

    if "```" in tests_code:
        parts = tests_code.split("```")
        for part in reversed(parts):
            if "def test" in part or "import pytest" in part:
                tests_code = part
                break

    if "import pytest" not in tests_code:
        tests_code = "import pytest\n\n" + tests_code

    test_path = write_test_file(tests_code, source_path)
    result = run_pytest_and_parse(str(Path(source_path).parent))
    result.update({
        "test_file": Path(test_path).name,
        "test_content": tests_code
    })
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gerar e rodar testes pytest usando Gemma via Transformers")
    parser.add_argument("source", help="Caminho para o arquivo Python a ser testado")
    args = parser.parse_args()
    print(f"Analisando arquivo: {args.source}")
    res = generate_tests_for_file(args.source)
    print("\n--- Conteúdo gerado pelo modelo ---")
    print(res.get("test_content", "Nenhum conteúdo gerado."))
    print("\n--- Relatório do programa ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nSucesso: {res.get('success', False)}")
