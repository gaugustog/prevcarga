---
marp: true
theme: default
size: 16:9
paginate: true
---

<!-- Título -->
<style>
section {
    --heading-strong-color: #FFFFF;
    padding-left: 150px !important;
    padding-top: 10px !important
}

h1 {
  font-size: 2em;
  color: #FFFFF; 
  margin-top: 100px;
}
h2 {
  font-size: 1.25em;
  color: #025159;
}
h3 {
  font-size: 1.em;
  color: #0e5d5a;
}
p, ul, li, td {
  font-size: .8em;
}

td {
  font-size: .625em;  
}

th {
  font-size: .8em;
}

.white-text {
    color: white;
}

.logo {
  position: absolute; /* Essencial para posicionamento livre */
  top: 50px;        /* Distância do topo */
  right: 50px;       /* Distância da direita */
}

.gtpw {
  position: absolute; /* Essencial para posicionamento livre */
  top: 25px;        /* Distância do topo */
  left: 150px;       /* Distância da direita */
}

.gpmh {
  position: absolute; /* Essencial para posicionamento livre */
  top: 25px;        /* Distância do topo */
  left: 280px;       /* Distância da direita */
}

.header-pointer {
  position: absolute; /* Essencial para posicionamento livre */
  top: 34px;        /* Distância do topo */
  left: 34px;       /* Distância da direita */
}



.sidebar-image {
  position: absolute; /* Essencial para posicionamento livre */
  top: 0px;        /* Distância do topo */
  left: 0px;       /* Distância da direita */
  bottom: 0px;
}

.new-badge {
  background-color: #4CAF50;
  color: white;
  padding: 2px 8px;
  text-align: center;
  border-radius: 12px;
  font-size: 0.7em;
  font-weight: bold;
  vertical-align: middle;
}
</style>

<!-- _backgroundImage: "url('assets/SlideBackgorund.jpg')" -->
<!-- _color: white -->
<img class="logo" src="assets/SlideLogo.svg" width="150px" />
<img class="gtpw" src="assets/SlideGPTW.png" width="110px" height="190px" />
<img class="gpmh" src="assets/SlideGPMH.png" width="110px" height="190px"/>


# **Desenvolvimento Cidadão no ONS**

## Uma Proposta de Transformação Estratégica para Agilidade e Inovação na Operação do Sistema Elétrico

**Auxiliando a Transformação Estratégica do ONS**
**Data:** 27 de Agosto de 2025

---

<!-- Agenda -->
<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />
<img class="header-pointer" src="assets/SlideHeaderPointer.png" width="280px" />

## Agenda

1.  **O Cenário Atual:** O Desafio da Entrega de TI
2.  **O que é Desenvolvimento Cidadão?**
3.  **Metodologia Ágil vs. Desenvolvimento Cidadão:** Uma Parceria
4.  **Além do Low-Code:** A Necessidade de uma Abordagem "Code-First"
5.  **Estratégia de Implementação e Governança:**
6.  **Governaça Tecnológica e a Filosofia Data Mesh**

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## 1. O Cenário Atual: O Desafio da Entrega de TI

Empresas em todo o mundo enfrentam **"sobrecargas nas equipes de TI"**.

*   **Demanda crescente:** A necessidade de novas aplicações, automações e análises de negócio cresce exponencialmente.
*   **Capacidade limitada:** As equipes de TI, mesmo as mais eficientes, possuem uma capacidade finita e um backlog de projetos que pode levar meses ou anos para ser atendido.
*   **O Risco do "Shadow IT":** Na ânsia por soluções, as áreas de negócio criam sistemas paralelos (planilhas complexas, bancos de dados em Access) sem governança, segurança ou escalabilidade.

**O Desenvolvimento Cidadão surge como uma solução para transformar o "Shadow IT" de um risco para uma inovação governada.**

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## 2. O que é Desenvolvimento Cidadão?

É uma abordagem que capacita **especialistas de domínio** — engenheiros, analistas, meteorologistas, etc. — a construir suas próprias soluções analíticas e de automação, utilizando plataformas e ferramentas **aprovadas e governadas pela TI**.

**O Desenvolvedor Cidadão no ONS é o especialista que:**
*   Entende profundamente do negócio (previsão de carga, análise de segurança elétrica, planejamento da operação).
*   Precisa de agilidade para testar hipóteses e criar novas análises.
*   Atualmente, muitas vezes depende de planilhas ou de longos ciclos de desenvolvimento da TI tradicional.

**Objetivo:** Mover a capacidade de desenvolvimento para mais perto de quem tem o conhecimento do problema, acelerando a inovação.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## 3. Método Ágil vs. Desenvolvimento Cidadão: Uma Parceria

A metodologia Ágil é essencial, mas não foi desenhada para resolver todos os tipos de problemas. O Desenvolvimento Cidadão atua como um complemento estratégico.

| Desafios do Modelo Ágil para o Negócio | Desenvolvimento Cidadão Complementa |
| :--- | :--- |
| **O "Gap de Tradução":** O especialista precisa traduzir seu requisito para um PO, que traduz para a TI. O conhecimento se dilui no processo. | O especialista **é o desenvolvedor**. A ideia e a implementação ocorrem na mesma mente, garantindo 100% de fidelidade ao requisito. |
| **Priorização no Backlog:** Soluções de nicho ou departamentais, mesmo que de alto impacto local, raramente são priorizadas no backlog global da TI. | **Autonomia para o Negócio.** A área de negócio pode desenvolver e implementar suas próprias soluções sem depender da fila de prioridades da TI. |
| **Velocidade de Experimentação:** Sprints de 2-4 semanas são lentos para a prototipação rápida e exploração de dados que a área de negócio necessita. | **Ciclos de horas ou dias.** Permite a criação de um MVP (Produto Mínimo Viável) de uma análise ou modelo em tempo recorde. |
| **Foco da TI Profissional:** O tempo dos desenvolvedores profissionais é valioso e deve ser focado em sistemas críticos, complexos e na infraestrutura core. | **Libera a TI para o Estratégico.** A TI foca em governar a plataforma e construir os "tijolos", e o negócio constrói as "casas". |

**Conclusão:** Não é **Ágil vs. Cidadão**, mas sim **Ágil + Cidadão** para uma empresa verdadeiramente adaptável.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## 4. Além do Low-Code: A Necessidade de uma Abordagem "Code-First"

Plataformas *Low-Code/No-Code* são excelentes para criar formulários, fluxos de aprovação e aplicativos simples.

**No entanto, para o ONS, elas são insuficientes pois não suportam:**
*   **Complexidade Algorítmica:** Modelos de otimização, previsão de vazões, machine learning para geração renovável e análise de estabilidade exigem código (Python, R, Julia).
*   **Reprodutibilidade Científica:** É vital que as análises sejam transparentes, auditáveis e reprodutíveis, o que é garantido pelo código-fonte versionado.
*   **Integração com Ferramentas de Alta Performance:** Necessidade de bibliotecas especializadas (ex: `pandas`, `scikit-learn`, `Pyomo`) e computação em larga escala.

Propomos uma abordagem **"Code-First"** ou **"Pro-Code" Citizen Development**: dar aos especialistas o poder de linguagens de programação completas, dentro de um ambiente seguro e governado.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## 5. Estratégia de Implementação e Governança

Uma implementação bem-sucedida não é sobre dar acesso irrestrito, mas criar uma "estrada pavimentada" para a inovação.

**Pilares da Estratégia:**

1.  **Centro de Habilitação (CoE):**
    *   Time multidisciplinar para definir padrões, treinar usuários e disseminar boas práticas.

2.  **Jornada do Desenvolvedor Cidadão:**
    *   **Nível 1 (Explorador):** Consome Produtos de Dados e cria dashboards.
    *   **Nível 2 (Desenvolvedor):** Cria e publica novos Produtos de Dados e modelos.
    *   **Nível 3 (Evangelista):** Cria soluções reutilizáveis e treina outros colegas.

3.  **Governança Federada & Alinhamento com Data Mesh:** 
    *   **TI Central:** Governa a **plataforma de dados self-service**, as **plataformas de BI e aplicação** (seja low-code ou pro-code), segurança, e os padrões para publicação de **Aplicações e Produtos de Dados**.
    *   **Domínios de Negócio (Citizen Developers):** Assumem a **posse (ownership)** dos dados e aplcações do seu domínio. São responsáveis por criar, manter e garantir a qualidade dos seus **Produtos de Dados** e **manutenção das Aplicações**.



---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## 6. Governança Tecnológica e a Filosofia Data Mesh

A arquitetura proposta materializa a filosofia Data Mesh, onde os Desenvolvedores Cidadãos são os donos dos produtos de dados.



---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.1 Colaboração e Versionamento: GitHub

**GitHub** como a plataforma central para o código-fonte de todos os Produtos de Dados e modelos.

*   **Git como Fonte da Verdade:** O código que define um produto de dados (validações, transformações, modelos) é versionado, auditável e reprodutível.
*   **Code Review entre Pares:** Garante a qualidade e a disseminação do conhecimento, reforçando a governança federada.
*   **Gestão de Projetos (Issues & Projects):** Transparência sobre a evolução e o ciclo de vida de cada produto de dados.
*   **Automação (GitHub Actions):** CI/CD para testar e implantar novas versões de um produto de dados de forma automática.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.1.1 Acelerando com IA: GitHub Copilot

GitHub Copilot é o componente que **reduz a barreira "Code-First"** para o Desenvolvedor Cidadão, atuando como um assistente de programação inteligente.

*   **De Intenção para Código:** O especialista de domínio descreve a lógica em linguagem natural (ex: `"# validar se a geração está dentro dos limites da usina"`) e o Copilot sugere o código em Python, traduzindo o conhecimento de negócio em lógica aplicável.
*   **Assistente de Codificação:** Autocompleta código repetitivo, sugere APIs de bibliotecas (ex: `pandas`) e acelera o desenvolvimento de scripts de validação, transformação e modelagem.
*   **Aumento da Qualidade:** Ajuda a gerar testes unitários para garantir a qualidade e a confiabilidade dos Produtos de Dados, um ponto crítico para a governança.
*   **Aprendizado Contínuo:** Funciona como uma ferramenta de aprendizado, expondo os Desenvolvedores Cidadãos a novas técnicas e boas práticas de programação.

---
<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.1.2 Inteligência Contextual: Como o Copilot Aprende com o ONS

A maior vantagem do **Copilot Business** é sua capacidade de **indexar o código privado dos repositórios do ONS** para fornecer sugestões altamente contextualizadas.

**Copilot Pro (Individual)**
*Sugestões baseadas em código público e no arquivo aberto.*
`Função genérica de validação`

**Copilot Business (ONS)**
*Sugestões baseadas em código público + **TODO o código privado do ONS.***
`Uso da classe ONSDataValidator com os métodos corretos.`

Isso significa que o Copilot passa a **"falar a língua do ONS"**, usando nossas APIs internas, padrões de projeto e bibliotecas como base para suas sugestões.

---
<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.1.3 Benefícios da Indexação: Qualidade, Velocidade e Governança

| Benefício | Impacto Direto para a TI e o Negócio |
| :--- | :--- |
| **Consistência e Padronização** | O Copilot sugere código usando as classes e funções internas do ONS, garantindo que as novas soluções sigam a arquitetura definida e evitando a "reinvenção da roda". |
| **Aceleração do Onboarding** | Um novo desenvolvedor pode perguntar ao Chat (`@workspace`): *"Como fazemos a autenticação neste serviço?"* e obter uma resposta baseada no nosso código, reduzindo a dependência de seniores. |
| **Governança Proativa** | As boas práticas são sugeridas **durante** a codificação, não apenas detectadas no code review. Isso melhora a qualidade desde o início e otimiza o tempo de revisão. |
| **Redução de Dívida Técnica** | Facilita a refatoração de código legado para os padrões modernos da organização, pois o Copilot já conhece os padrões mais novos usados em outros repositórios. |

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.2 Validação e Criação de Produtos de Dados 

**Proposta:** Plataforma de Validação em contêineres para transformar dados brutos em **Produtos de Dados Confiáveis**.

1.  **Desenvolvimento (GitHub):** O Desenvolvedor Cidadão (dono do domínio) codifica as regras de validação e transformação em Python.
2.  **Conteinerização:** A lógica é empacotada em um contêiner Docker.
3.  **Execução (AWS Fargate):** Um gatilho (ex: novo arquivo) inicia a execução do contêiner.
4.  **Resultado e Publicação no Data Mesh:**
    *   **Dados Válidos:** São publicados no **Catálogo do Data Mesh** como um **Produto de Dados** certificado, com metadados claros (dono, linhagem, qualidade, SLA). Agora, está disponível para todo o ONS.
    *   **Dados Inválidos:** Um relatório detalhado é enviado ao fornecedor, fortalecendo o ecossistema.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.3 Modelagem como um Produto de Dados Derivado 

Modelos consomem e produzem produtos de dados, vivendo dentro do Mesh.

**Proposta:** Plataforma de MLOps integrada ao Data Mesh.

*   **Consumo de Dados:** Os modelos são treinados utilizando **Produtos de Dados** confiáveis do Mesh (ex: "telemetria validada de usinas", "previsão meteorológica oficial").
*   **Jobs Orquestrados:**
    1.  **Input:** Busca de Produtos de Dados no Mesh.
    2.  **Treinamento, Inferência, Combinação.**
    3.  **Output (Publicação):** O resultado do modelo (ex: "previsão de carga horária para o Sul") é publicado como um **novo Produto de Dados, derivado** e disponível no Mesh para outras áreas.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

### 6.4 Visualização e Entrega: Posit Connect

**Posit Connect** como a camada de "vitrine" do Data Mesh, permitindo o consumo fácil dos produtos de dados.

*   **Publicação com 1 Clique:** Desenvolvedores Cidadãos criam aplicações que consomem dados do Mesh e as publicam como:
    *   Dashboards interativos (Streamlit, Dash, Shiny).
    *   APIs REST para integrações (ex: `api/previsao_carga/sul`).
    *   Relatórios agendados.
*   **Acesso Governado:** Garante que os produtos de dados e as análises sejam consumidos apenas por quem tem permissão.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## Próximos Passos e Conclusão

**O Desenvolvimento Cidadão, alinhado à estratégia de Data Mesh, transforma a TI de um centro de custo em uma plataforma habilitadora de inovação descentralizada.**

**Proposta de Ação:**
1.  **Projeto Piloto (Prova de Conceito):**
    *   Selecionar 1 caso de uso para criar o **primeiro Produto de Dados de ponta a ponta**.
    *   Exemplo: "Produto de Dados de Geração Verificada da Usina X".
    *   Time: 2-3 especialistas do domínio (futuros donos do produto) + 1-2 arquitetos de TI (habilitadores da plataforma).
    *   **Meta:** Em 3 meses, ter o produto publicado no Data Mesh e sendo consumido por um relatório no Posit Connect.

2.  **Estruturação do Centro de Habilitação (CoE):**
    *   Focar em definir os padrões para "um bom produto de dados" e treinar os times na nova filosofia.

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## Perguntas e Discussão

---

<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## Resumo Estratégico: Riscos e Valor

### Maiores Riscos e Mitigações

*   **Adoção Cultural:** Resistência à mudança.
    *   **Mitigação:** Patrocínio executivo forte, comunicação clara dos benefícios e começar com voluntários (early adopters).
*   **Governança Fraca:** Criação de um "pântano de dados" em vez de um Mesh.
    *   **Mitigação:** CoE atuante, padrões de qualidade automatizados e ownership claro dos produtos de dados.
*   **Complexidade da Ferramenta:** Curva de aprendizado de "Code-First".
    *   **Mitigação:** Treinamento contínuo, templates, suporte intensivo e uso de aceleradores como IA.

### Valor Agregado para o ONS

*   **Agilidade Operacional:** Respostas rápidas a eventos do SIN.
*   **Inovação na Ponta:** Soluções criadas por quem mais entende do problema.
*   **Redução de Risco:** Fim do "Shadow IT" e aumento da segurança e auditoria.
*   **Eficiência da TI:** Foco da TI no que é mais estratégico para a empresa.
*   **Retenção de Talentos:** Empoderamento e desenvolvimento dos especialistas de domínio.

---
<img class="sidebar-image" src="assets/LeftSidebarSlide.jpg" width="120px" />

## Alinhamento Estratégico com a TI: Abordando Preocupações Comuns

| Objeção Comum | Nossa Abordagem (Governança Centralizada, Execução Federada) |
| :--- | :--- |
| **"Isso é Shadow IT com outro nome. Perderemos o controle e criaremos o caos."** | Pelo contrário. **Trazemos o Shadow IT para a luz**. Em vez de planilhas e scripts soltos, as soluções nascem em uma plataforma **controlada e observável pela TI**. |
| **"Usuários de negócio não são desenvolvedores. Vão criar código inseguro e de baixa qualidade."** | Concordamos. Por isso a abordagem é **"Code-First, mas em uma estrada pavimentada"**. A TI provê os templates, as bibliotecas seguras, os pipelines de CI/CD e as políticas de segurança. O CoE treina e audita. |
| **"Quem vai dar suporte a tudo isso? Vai sobrecarregar a TI."** | O modelo de **ownership do Data Mesh** é a resposta. O domínio de negócio é dono do produto de dados que cria, incluindo o suporte funcional (N1/N2). A TI suporta a **plataforma**, não as centenas de aplicações. |
| **"Isso vai aumentar os custos e a complexidade da nossa arquitetura."** | O custo do **não fazer** (oportunidades perdidas no backlog, riscos do Shadow IT) é maior. A arquitetura proposta **centraliza e simplifica** o acesso a dados e a publicação, reduzindo a complexidade atual de integrações ponto a ponto. |
