
# Nom du Projet

**Conception et développement d’un chatbot transactionnel intelligent**

---

## Table des matières

1. [Description du Projet](#description-du-projet)
2. [Fonctionnalités](#fonctionnalités)
3. [Technologies Utilisées](#technologies-utilisées)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
6. [Architecture du Projet](#architecture-du-projet)
7. [Contributeurs](#contributeurs)
8. [Contact](#contact)

---

## Description du Projet

Ce projet vise à concevoir et développer un chatbot transactionnel intelligent pour une application bancaire. L'objectif est de fournir une solution automatisée qui puisse interagir avec les utilisateurs et répondre à leurs demandes, telles que la consultation de soldes ou l'exécution de virements. Le modèle de traitement du langage naturel est basé sur **BERT**, afin d'assurer une compréhension précise des demandes des utilisateurs.

## Fonctionnalités

- **Consulter le Solde** : Demandez votre solde en utilisant des phrases naturelles.
- **Virement Mandat** : Réalisez un virement mandat simplement en donnant la commande au chatbot.
- **Consulter CCP** : Vérifiez les informations sur votre compte CCP.
- **Virement CCP** : Effectuez un virement à partir de votre compte CCP.

## Technologies Utilisées

- **Python** : Pour la logique du projet.
- **BERT** : Modèle de traitement du langage naturel pour le chatbot.
- **FastAPI** : Pour la création de l'API de backend.
- **React** : Pour l'interface utilisateur.
- **Ollama** : Pour déployer le modèle LLM.
- **Docker** : Pour containeriser l'application et faciliter le déploiement.

## Installation

Pour cloner et exécuter ce projet localement, suis ces étapes :

1. Clonez le dépôt GitHub :
    ```sh
    git clone https://github.com/ton-utilisateur/ton-projet.git
    ```
2. Naviguez dans le répertoire du projet :
    ```sh
    cd ton-projet
    ```
3. Installez les dépendances Python dans un environnement virtuel :
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    pip install -r requirements.txt
    ```

## Utilisation

Pour lancer le projet localement :

1. Démarrez l'API FastAPI :
    ```sh
    uvicorn server.main:app --reload
    ```
2. Naviguez vers l'interface React :
    ```sh
    cd client
    npm start
    ```
3. Accédez à l'application à l'adresse `http://localhost:3000`.

## Architecture du Projet

- **Dossier `server/`** : Contient le code backend, y compris les modèles et l'API FastAPI.
- **Dossier `client/`** : Inclut l'interface utilisateur construite avec React.
- **Dossier `models/`** : Contient les modèles pré-entraînés et ceux finement ajustés.
- **`Dockerfile`** : Pour la création d'images Docker du projet.
- **`requirements.txt`** : Liste des dépendances nécessaires.

## Contributeurs

- **Samar Ammar** - [Profil GitHub]([https://github.com/ton-utilisateur](https://github.com/samaraxmmar))

## Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter :

- **Email** : [samarammar070@gmail.com)
