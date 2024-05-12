# Oi, sou Juliena Alves, criadora deste simulador de entrevistas :)
# Decici criar um, por mais que fosse ainda algo simples, pois estou a procura de emprego e praticar entrevistas é muito importante.
# E assim como eu, muitos estão nessa busca também. Então porque não criar uma ferramenta que ajude nesse momento?

# Abaixo, adicionei comentários no código de forma a dar um pouco mais de detalhes de seu funcionamento
# Este programa é primeira versão de um chatbot que simula uma entrevista de emprego com o usuário final. Criado com os conhecimentos que adiquiri
# durante a semana de imersão Alura com Google

# Sinta-se à vontade para dar feedback construitivo e sugestões para que o programa se torne mais robusto e amplo


# Pra começar, uma explicação simples e direta de cada biblioteca importada no programa e seu objetivo:

# 1. google.generativeai:
# Objetivo: Permite usar a inteligência artificial generativa do Google, como o modelo Gemini, para gerar textos criativos e responder perguntas.

# 2. nltk (Natural Language Toolkit):
# Objetivo: Fornece ferramentas para processamento de linguagem natural (PLN), como como tokenização, radicalização e análise sintática.

# 3. nltk.tokenize:
# Objetivo: Divide textos em unidades menores, como palavras ou frases, para facilitar a análise.

# 4. nltk.stem:
# Objetivo: Reduz palavras ao seu radical, ignorando variações gramaticais, para facilitar a comparação e análise.

# 5. nltk.tag:
# Objetivo: Analisa a estrutura gramatical de frases, identificando a função de cada palavra (substantivo, verbo, adjetivo, etc.).

# 6. requests:
# Objetivo: Permite fazer requisições HTTP para interagir com servidores web e APIs, como a API do Google AI.

# 7. google.colab:
# Objetivo: Fornece funções específicas para o ambiente Google Colab, como armazenamento seguro de dados do usuário.

# Em resumo: O programa usa a IA do Google para simular uma entrevista de emprego. 
# Ele processa o texto das perguntas e respostas usando técnicas de PLN, analisa a linguagem e 
# gera feedback personalizado com base no conteúdo e nas habilidades da vaga escolhida.

# Antes de ir pro código, você pode estar se perguntando o que é tokenização, radicalização e análise sintática. Vamos lá!

'''
Imagine que você tem um texto e quer que o computador o entenda. Para isso, precisamos "quebrar" esse texto em partes menores e 
analisar cada uma delas. É aí que entram a tokenização, radicalização e análise sintática:

1. Tokenização:
O que é: É como cortar um bolo em fatias. Pegamos um texto e o dividimos em unidades menores, chamadas tokens.
Exemplo: A frase "O rato roeu a roupa do rei de Roma." seria dividida nos seguintes 
tokens: ["O", "rato", "roeu", "a", "roupa", "do", "rei", "de", "Roma", "."].
Para que serve: Facilita a análise do texto, permitindo que o computador trabalhe com cada palavra ou símbolo separadamente.

2. Radicalização (Stemming):
O que é: É como encontrar a "raiz" de uma palavra, ignorando suas variações gramaticais.
Exemplo: As palavras "correndo", "correu" e "correrão" teriam o mesmo radical: "corr".
Para que serve: Agrupa palavras semelhantes, facilitando a comparação e a busca por termos relacionados, 
mesmo que estejam escritos de formas diferentes.

3. Análise Sintática (POS Tagging):
O que é: É como identificar a função de cada palavra em uma frase, como se estivéssemos fazendo uma análise gramatical.
Exemplo: Na frase "O gato preto pulou a cerca.", a análise sintática identificaria "gato" como substantivo, 
"preto" como adjetivo, "pulou" como verbo e "cerca" como substantivo.
Para que serve: Entender a estrutura da frase e a relação entre as palavras, permitindo que o computador interprete o 
significado do texto de forma mais precisa.

Em resumo:
Essas três técnicas são como ferramentas que ajudam o computador a "dissecar" um texto, entendendo suas partes, 
relacionando palavras e interpretando seu significado de forma mais completa.

'''

# Agora vamos ao que interessa! 

# Import Model
import google.generativeai as genai # Importa a biblioteca do Google para IA Generativa
# Imports para uso de técnicas PLN
import nltk # Importa a biblioteca NLTK para Processamento de Linguagem Natural
from nltk.tokenize import word_tokenize # Importa a função para tokenizar texto
from nltk.stem import PorterStemmer # Importa o stemmer Porter para reduzir palavras ao seu radical
from nltk.tag import pos_tag # Importa a função para realizar a análise sintática (POS tagging)
# Import request
import requests # Importa a biblioteca para fazer requisições HTTP
# Used to securely store your API key
from google.colab import userdata # Importa a função para acessar dados do usuário no Google Colab
api_key = userdata.get("SECRET_KEY") # Obtém a chave de API armazenada no ambiente do usuário
genai.configure(api_key=api_key) # Configura a IA Generativa do Google com a chave de API

nltk.download('punkt') # Faz o download dos dados necessários para tokenização
nltk.download('averaged_perceptron_tagger') # Faz o download dos dados para POS tagging

generation_config = {
  "candidate_count":1, # Define que apenas uma resposta será gerada
  "temperature": 0.5 # Define a "criatividade" da IA (valores mais altos geram respostas mais criativas)
}

model = genai.GenerativeModel(model_name = "gemini-1.0-pro", # Define o modelo de IA a ser usado
                              generation_config = generation_config) # Define as configurações de geração

chat = model.start_chat(history=[]) # Inicia um chat com a IA Generativa

habilidades_tecnicas = { # Dicionário que relaciona cargos com suas habilidades técnicas
    "Desenvolvedor Python": ["Python", "Django", "Flask", "SQL", "Git"],
    "Analista de Dados": ["R", "SQL", "Pandas", "Machine Learning", "Tableau"],
    "Designer UX/UI": ["Figma", "Adobe XD", "Sketch", "UI/UX Principles", "User Research"],
}

habilidades_comportamentais = { # Dicionário que relaciona cargos com suas habilidades comportamentais
    "Desenvolvedor Python": ["Comunicação clara", "Resolução de problemas", "Aprendizagem contínua", "Trabalho em equipe", "Gerenciamento de tempo"],
    "Analista de Dados": ["Orientação para detalhes", "Pensamento analítico", "Habilidades de comunicação", "Criatividade", "Adaptabilidade"],
    "Designer UX/UI": ["Empatia", "Criatividade", "Comunicação visual", "Resolução de problemas", "Trabalho em equipe"],
}

def gerar_feedback(resposta, habilidade, vaga): # Função para gerar feedback para o usuário
    # Pré-processamento
    resposta_limpa = resposta.lower().replace(".", "").strip() # Limpa a resposta do usuário, removendo pontuação e espaços em branco
    habilidade_limpa = " ".join(habilidade).lower().replace(".", "").strip() # Limpa a lista de habilidades, removendo pontuação e espaços em branco
    resposta_tokenizada = word_tokenize(resposta_limpa) # Tokeniza a resposta do usuário
    habilidade_tokenizada = word_tokenize(habilidade_limpa) # Tokeniza a lista de habilidades

    # Análise sintática
    resposta_pos_tags = pos_tag(resposta_tokenizada) # Realiza a análise sintática da resposta do usuário
    habilidade_pos_tags = pos_tag(habilidade_tokenizada) # Realiza a análise sintática da lista de habilidades

    # Análise semântica
    pontos_chave = {}  # Dicionário para armazenar pontos fortes e fracos
    for palavra_chave, tag in habilidade_pos_tags: # Itera sobre cada palavra-chave na lista de habilidades
        if palavra_chave in resposta_tokenizada: # Verifica se a palavra-chave está presente na resposta do usuário
            pontos_chave[palavra_chave] = "forte" # Se estiver, marca como ponto forte
        else:
            pontos_chave[palavra_chave] = "fraco" # Se não estiver, marca como ponto fraco

    # Geração de feedback
    feedback = f"Sua resposta: {resposta}\n\n" # Inicia a construção do feedback

    if pontos_chave: # Se houver pontos_chave identificados
        pontos_fortes = [chave for chave, valor in pontos_chave.items() if valor == "forte"] # Cria uma lista com os pontos fortes
        if pontos_fortes: # Se houver pontos fortes
            feedback += f" Para a vaga escolhida, você declarou conhecimento em: {', '.join(pontos_fortes)}\n\n" # Adiciona os pontos fortes ao feedback

        pontos_fracos = [chave for chave, valor in pontos_chave.items() if valor == "fraco"] # Cria uma lista com os pontos fracos
        if pontos_fracos: # Se houver pontos fracos
            feedback += f" Contudo, para a vaga de {vaga}, seria interessante demonstrar conhecimento em: {', '.join(pontos_fracos)}\n\n" # Adiciona os pontos fracos ao feedback
            feedback += " Que tal se aprofundar nesses tópicos? Aqui estão algumas sugestões de cursos:\n\n" # Sugere que o usuário se aprofunde nos tópicos fracos

            for palavra_chave in pontos_fracos: # Itera sobre cada ponto fraco
                response = model.generate_content(f"Liste 3 cursos online sobre {palavra_chave}") # Usa a IA para gerar sugestões de cursos online sobre o ponto fraco
                if response: # Se houver resposta da IA
                    feedback += f"{response.text}\n\n" # Adiciona as sugestões de cursos ao feedback

    # Sugestão de resposta com base nas habilidades da vaga
    feedback += "**Que tal tentar esta resposta?**\n\n" # Sugere uma resposta completa para o usuário
    response = model.generate_content(f"Crie uma resposta para a pergunta, mencionando as seguintes habilidades: {', '.join(habilidade)}") # Usa a IA para gerar uma resposta completa com base nas habilidades da vaga
    if response: # Se houver resposta da IA
        feedback += f"{response.text}\n\n" # Adiciona a resposta completa ao feedback

    feedback += "Lembre-se: Adapte sua resposta de acordo com as necessidades específicas da vaga e da empresa, além da sua própria experiência." # Recomenda que o usuário personalize a resposta

    return feedback # Retorna o feedback completo

def simular_entrevista(vaga): # Função que simula a entrevista com base na vaga escolhida
    # Definir perguntas de acordo com a vaga
    if vaga == "Desenvolvedor Python": # Define as perguntas para a vaga de Desenvolvedor Python
        perguntas = [
            "Me fale sobre sua experiência como Desenvolvedor Python.",
            "Qual projeto Python do qual você mais se orgulha?",
            "Como você resolveria um erro de codificação inesperado?",
            "Como você se mantém atualizado com as últimas tecnologias em desenvolvimento?",
            "Conte-me sobre um desafio que você superou em um projeto de desenvolvimento.",
        ]
        habilidade = habilidades_tecnicas[vaga] # Define as habilidades técnicas para a vaga
    elif vaga == "Analista de Dados": # Define as perguntas para a vaga de Analista de Dados
        perguntas = [
            "Descreva sua experiência com análise de dados.",
            "Qual ferramenta de análise de dados você mais utiliza?",
            "Como você comunicaria os resultados de uma análise de dados para stakeholders não técnicos?",
            "Descreva um projeto de análise de dados que você realizou e qual foi o impacto do projeto.",
            "Como você se manteria atualizado com as últimas tendências em análise de dados?",
        ]
        habilidade = habilidades_tecnicas[vaga] # Define as habilidades técnicas para a vaga
    elif vaga == "Designer UX/UI": # Define as perguntas para a vaga de Designer UX/UI
        perguntas = [
            "Descreva sua experiência com design UX/UI.",
            "Qual software de design UX/UI você mais utiliza?",
            "Como você realiza a pesquisa do usuário para seus projetos?",
            "Descreva um projeto de design UX/UI que você realizou e quais foram os principais desafios.",
            "Como você se manteria atualizado com as últimas tendências em design UX/UI?",
        ]
        habilidade = habilidades_tecnicas[vaga] # Define as habilidades técnicas para a vaga
    else: # Define perguntas genéricas para outras vagas
        perguntas = ["Descreva sua experiência profissional.",
                     "Quais são seus pontos fortes e fracos como profissional?",
                     "Por que você se interessa por essa vaga?",
        ]
        habilidade = ["Experiência profissional", "Pontos fortes", "Pontos fracos", "Interesse na vaga"] # Define habilidades genéricas para outras vagas

    for pergunta in perguntas: # Itera sobre cada pergunta
        print("Recrutador(a): " + pergunta + "\n> ") # Imprime a pergunta
        resposta = input("Digite sua resposta: " + "\n> ")  # Recebe a resposta do usuário

        feedback = gerar_feedback(resposta, habilidade, vaga) # Gera o feedback com base na resposta, habilidades e vaga
        print(feedback + "\n> ")  # Imprime o feedback

    return simular_entrevista # Retorna a função para permitir a simulação de outra entrevista


print("Iniciando o simulador de entrevista...." + "\n> ")
vaga_escolhida = prompt = input("Primeiro informe uma vaga que deseja se candidatar:  " + "\n> ") # Recebe a vaga desejada pelo usuário
print("Vaga Escolhida: " + vaga_escolhida + "\n> ") # Imprime a vaga escolhida

simular_entrevista = simular_entrevista(vaga_escolhida) # Inicia a simulação da entrevista com base na vaga escolhida