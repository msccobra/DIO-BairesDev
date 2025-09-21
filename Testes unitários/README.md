# Gerador Automático de Testes com Gemma

Este programa gera automaticamente testes unitários para arquivos Python usando inteligência artificial em um modelo rodando localmente, ao invés de usar diretamente uma solução na nuvem como Azure Open AI, que seria o ideal, mas é uma solução dependente de uma assinatura ativa na plataforma. Como a minha demonstração já expirou, eu optei por fazer uma alternativa grátis, que roda tudo localmente. O modelo escolhido foi o Gemma-2-2b, desenvolvido pelo Google.

Esse modelo permite o programa rodar localmente sem um custo exacerbado de hardware, e no programa foi feita a divisão de tarefas entre a CPU e a GPU, permitindo que o modelo rode em máquinas com, menos de 16GB de RAM, usando ou não aceleração via GPU. Nesse último caso, como a carga na RAM é menor, é possível usar o computador normalmente para tarefas leves paralelamente ao uso do programa.

## Como funciona?
- Você fornece um arquivo Python como entrada.
- O programa usa o modelo Gemma para criar testes no formato pytest.
- Os testes são salvos em um arquivo chamado `test_generated.py` na mesma pasta do seu código.
- O programa executa os testes e mostra um relatório com os resultados.

## Principais recursos
- Geração automática de testes para funções públicas do seu código.
- Suporte a execução em CPU ou GPU (se disponível).
- Relatório detalhado: mostra quantos testes passaram, falharam ou tiveram erro.
- Se o modelo não gerar testes válidos, o programa cria um esqueleto básico de testes.

## Como usar
1. Configure o arquivo `.env` com seu token do Hugging Face:
   ```
   HUGGINGFACE_HUB_TOKEN=seu_token_aqui
   ```
2. Execute o programa no terminal:
   ```
   python agent_gemma_hf7.py caminho/do/seu/arquivo.py
   ```
3. Veja o arquivo `test_generated.py` gerado e o relatório no terminal.

## Requisitos
- Python 3.10+
- PyTorch
- Transformers
- pytest
- Conta no Hugging Face

## Dica
Você pode usar este programa para automatizar a criação de testes em projetos Python, economizando tempo e garantindo mais qualidade no seu código.

