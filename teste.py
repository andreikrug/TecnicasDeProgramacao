import ollama
from ollama import chat
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Carregando spaCy...")
nlp = spacy.load("pt_core_news_lg")

print("Carregando modelo de embeddings...")
embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

# Schemas
# proximos passos seria recuperar os schemas do database, pegar 1 valor de coluna, talvez explicar o que cada coluna armazena com exemplos


SCHEMAS = [
    {
        "id": "chamados",

        "texto": """
        TABELA chamados

        DESCRIÇÃO:
        Registra chamados de ouvidoria.

        COLUNAS:
        protocolo
        descricao
        categoria
        bairro
        statussolicitacao
        dataregistro
        datafinalprevista
        dataenvioresposta
        tipofim
        ano

        EXEMPLOS:
        falta de iluminação
        descricao = iluminação
        buraco na rua
        coleta de lixo
        ano = 2025
        """
    },

    {
        "id": "usuarios",

        "texto": """
        TABELA usuarios

        DESCRIÇÃO:
        Usuários responsáveis pelos atendimentos.

        COLUNAS:
        id
        nome
        email
        setor
        """
    }
]

# gerar embedings do schema

print("Gerando embeddings dos schemas...")

for schema in SCHEMAS:

    schema["embedding"] = embedding_model.encode(
        schema["texto"]
    )

# NLP - Tokenizar e lematizar a pergunta do usuario

def analisar_pergunta(pergunta):

    doc = nlp(pergunta)

    print("\n===== TOKENS =====")

    for token in doc:

        print(
            token.text,
            "| lema:",
            token.lemma_,
            "| classe:",
            token.pos_
        )

    print("\n===== LEMMAS =====")

    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
    ]

    print(lemmas)

    print("\n===== ENTIDADES =====")

    entidades = []

    for ent in doc.ents:

        entidade = {
            "texto": ent.text,
            "tipo": ent.label_
        }

        entidades.append(entidade)

        print(
            ent.text,
            "->",
            ent.label_
        )

    return {
        "lemmas": lemmas,
        "entidades": entidades
    }

# Montar prompt que sera integrado com modelo qwen

def construir_prompt(
    pergunta,
    schema,
    analise
):

    prompt = f"""
PERGUNTA DO USUÁRIO:

{pergunta}

INFORMAÇÕES EXTRAÍDAS:

Lemmas:
{analise['lemmas']}

Entidades:
{analise['entidades']}

SCHEMA DISPONÍVEL:

{schema['texto']}

INSTRUÇÕES:

- Utilize apenas tabelas existentes.
- Utilize apenas colunas existentes.
- Gere apenas SQL.
- Não explique o raciocínio.
- Pode inferir tabela e coluna caso não ache a exata
- Não leve mais de 2 minutos para retornar a resposta
- Sempre retorne alguma resposta

SQL:
"""

    return prompt
# Recuperação semantica - Comparar embeddings da pergunta do usuario com as embeddings de cada esquema

def buscar_schema_relevante(pergunta):

    pergunta_embedding = embedding_model.encode(
        pergunta
    )

    melhor_schema = None
    melhor_score = -1

    print("\n===== SIMILARIDADE =====")

    for schema in SCHEMAS:

        score = cosine_similarity(
            [pergunta_embedding],
            [schema["embedding"]]
        )[0][0]

        print(
            f"{schema['id']} -> {score:.4f}"
        )

        if score > melhor_score:

            melhor_score = score
            melhor_schema = schema

    return melhor_schema, melhor_score



# Pipeline da execeução principal

def pipeline():

    pergunta = input(
        "\nDigite sua pergunta:\n> "
    )

    print("\n===== ETAPA 1 - NLP =====")

    analise = analisar_pergunta(
        pergunta
    )

    print("\n===== ETAPA 2 - RECUPERAÇÃO =====")

    schema, score = buscar_schema_relevante(
        pergunta
    )

    print("\n===== RESULTADO =====")

    print(
        f"Schema mais relevante: {schema['id']}"
    )

    print(
        f"Similaridade: {score:.4f}"
    )

    print("\n===== SCHEMA RECUPERADO =====")

    print(schema["texto"])

    print("\n===== RESUMO =====")

    print("Lemmas encontrados:")
    print(analise["lemmas"])

    print("\nEntidades encontradas:")
    print(analise["entidades"])
    

    prompt = construir_prompt(
        pergunta,
        schema,
        analise
    )

    print(prompt)

    response = ollama.chat(model='qwen3.5:2b', messages=[
    {
        'role': 'system',
        'content': 'Você é um especialista em SQL. Sua tarefa é responder perguntas utilizando apenas as tabelas e colunas fornecidas.',
    },
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    print(response['message']['content'])

if __name__ == "__main__":

    pipeline()
