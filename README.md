# Central CFTV

Sistema local de monitoramento por camera com interface web, deteccao de movimento, gravacao automatica, snapshots, zonas de deteccao, agenda de funcionamento, notificacoes e suporte a multiplas cameras.

O projeto roda em Python com Flask e OpenCV. Ao iniciar, ele abre um servidor local em `http://127.0.0.1:5051`.

## Funcionalidades

- Visualizacao ao vivo da camera via navegador.
- Suporte a cameras IP por URL de video e cameras locais por indice, como `0` ou `camera:0`.
- Cadastro de multiplas cameras.
- Deteccao de movimento com OpenCV.
- Ajuste de sensibilidade e calibracao automatica.
- Zonas de deteccao configuraveis.
- Captura manual de foto.
- Foto automatica ao detectar movimento.
- Gravacao automatica ao detectar movimento, com pre-gravacao e pos-gravacao.
- Agenda de ativacao por horario.
- Controle manual e automatico de flash para cameras IP compativeis.
- Notificacoes no navegador.
- Envio de eventos para webhook.
- Armazenamento local de configuracoes, eventos, snapshots e gravacoes.

## Tecnologias

- Python
- Flask
- OpenCV
- NumPy
- PyInstaller

## Requisitos

- Python 3.10 ou superior
- Camera IP com stream de video, ou uma webcam/camera local
- Navegador moderno

## Instalacao

Clone o repositorio:

```bash
git clone https://github.com/system-auto-jk/CentralCFTV.git
cd CentralCFTV
```

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependencias:

```bash
pip install flask opencv-python numpy pyinstaller
```

## Como executar

Rode o sistema:

```bash
python app.py
```

Depois acesse:

```text
http://127.0.0.1:5051
```

## Configurando cameras

Na interface web, use o campo de fonte de video para informar:

- Camera IP: `http://IP_DA_CAMERA:PORTA/video`
- Camera local: `0`
- Camera local alternativa: `camera:0`

O projeto vem com uma URL padrao de exemplo em `app.py`:

```text
http://192.168.1.71:8080/video
```

Altere pela interface ou edite o arquivo `security_data/settings.json` depois da primeira execucao.

## Dados gerados

O sistema cria e usa a pasta `security_data` para armazenar dados locais:

```text
security_data/
  settings.json
  events.json
  snapshots/
  recordings/
```

Arquivos importantes:

- `settings.json`: configuracoes das cameras e preferencias.
- `events.json`: historico de eventos de movimento.
- `snapshots/`: fotos capturadas.
- `recordings/`: gravacoes em video.

## Gerar executavel

O projeto possui o arquivo `CentralCFTV.spec` para empacotar com PyInstaller.

Para gerar o executavel:

```bash
pyinstaller CentralCFTV.spec
```

O executavel sera criado em:

```text
dist/CentralCFTV.exe
```

Ao executar o `.exe`, acesse o navegador em:

```text
http://127.0.0.1:5051
```

## Estrutura principal

```text
.
|-- app.py
|-- CentralCFTV.spec
|-- security_data/
`-- README.md
```

## Preparando para publicar no GitHub

Antes de publicar, e recomendado nao versionar arquivos gerados automaticamente, como:

```text
build/
dist/
__pycache__/
*.log
*.pyc
security_data/snapshots/
security_data/recordings/
```

Se quiser manter exemplos de configuracao, voce pode deixar apenas arquivos JSON de exemplo, sem dados reais de cameras, IPs privados ou historico de eventos.

## Observacoes de seguranca

- O servidor Flask esta configurado para rodar apenas localmente em `127.0.0.1`.
- Nao publique URLs privadas de cameras, webhooks ou registros reais de monitoramento.
- Revise `security_data/settings.json` e `security_data/events.json` antes de subir o projeto para um repositorio publico.

## Licenca

Defina uma licenca antes de publicar o projeto. Uma opcao comum para projetos abertos e a licenca MIT.
