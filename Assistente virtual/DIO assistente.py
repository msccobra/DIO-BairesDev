import speech_recognition as sr
from gtts import gTTS
import os
from datetime import datetime
from playsound import playsound
import webbrowser

def audio():
    voz = sr.Recognizer()
    with sr.Microphone() as source:
        voz.adjust_for_ambient_noise(source, duration=1)
        gTTS("Qual seu comando?", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/diga.mp3")
        playsound(r"C:/Users/flawl/Music/audios/diga.mp3")
        print("Qual seu comando?")
        audio = voz.listen(source)
        try:
            texto = voz.recognize_google(audio, language='pt-BR')
            print("Você disse: " + texto)
        except sr.UnknownValueError:
            print("Não entendi")
            pass
    return texto.lower()

def comando(texto): #função que permite a execução de alguns comandos simples
    if "que horas são" in texto:
        agora = datetime.now()
        horas = agora.strftime("%H:%M")
        print("Agora são " + horas)
        gTTS("Agora são " + horas, lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/hora.mp3")
        playsound(r"C:/Users/flawl/Music/audios/hora.mp3")
    elif "abra o youtube" in texto:
        gTTS('O que você quer assistir?', lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/youtube.mp3")
        playsound(r"C:/Users/flawl/Music/audios/youtube.mp3")
        busca = sr.Recognizer()
        with sr.Microphone() as source:
            busca.adjust_for_ambient_noise(source, duration=1)
            print("O que você quer assistir?")
            audio = busca.listen(source)
            try:
                busca = busca.recognize_google(audio, language='pt-BR')
                print("Você disse: " + busca)
            except sr.UnknownValueError:
                print("Não entendi")
                busca = ""
                pass
        if busca != "":
            url = f"https://www.youtube.com/results?search_query=" + busca
            gTTS("Aqui está o que eu encontrei sobre " + busca + " no YouTube", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/busca.mp3")
            playsound(r"C:/Users/flawl/Music/audios/busca.mp3")
            webbrowser.open(url)
    elif "abra o google" in texto:
        gTTS('O que você quer pesquisar?', lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/google.mp3")
        playsound(r"C:/Users/flawl/Music/audios/google.mp3")
        busca = sr.Recognizer()
        with sr.Microphone() as source:
            busca.adjust_for_ambient_noise(source, duration=1)
            print("O que você quer pesquisar?")
            audio = busca.listen(source)
            try:
                busca = busca.recognize_google(audio, language='pt-BR')
                print("Você disse: " + busca)
            except sr.UnknownValueError:
                print("Não entendi")
                busca = ""
                pass
        if busca != "":
            url = f"www.google.com/search?q=" + busca
            gTTS("Aqui está o que eu encontrei sobre " + busca + " no Google", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/busca.mp3")
            playsound(r"C:/Users/flawl/Music/audios/busca.mp3")
            webbrowser.open(url)
    elif "abra o gmail" in texto:
        print("Abrindo o Gmail")
        gTTS("Abrindo o Gmail", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/gmail.mp3")
        playsound(r"C:/Users/flawl/Music/audios/gmail.mp3")
        webbrowser.open("https://mail.google.com/mail/u/0/#inbox")
    elif "abra o steam" in texto:
        print("Abrindo o Steam") # esse foi um capricho meu :)
        gTTS("Abrindo o Steam", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/steam.mp3")
        playsound(r"C:/Users/flawl/Music/audios/steam.mp3")
        os.startfile(r"C:\Program Files (x86)\Steam\steam.exe")
    elif "abra o deezer" in texto:
        print("Abrindo o Deezer")
        gTTS("Abrindo o Deezer", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/deezer.mp3")
        playsound(r"C:/Users/flawl/Music/audios/deezer.mp3")
        webbrowser.open("https://www.deezer.com/br/")
    else:
        print("Comando não reconhecido")
        gTTS("Comando não reconhecido", lang='pt-br', tld = 'com.br').save(r"C:/Users/flawl/Music/audios/nada.mp3")
        playsound(r"C:/Users/flawl/Music/audios/nada.mp3")

comando(audio())