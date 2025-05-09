Analisando seus arquivos de configuração, identifiquei o problema principal e algumas recomendações para corrigir o comportamento indesejado do bot:
Problema Principal

O bot está sempre acionando o fallback (action_fallback_web_search) porque:

    Configuração do FallbackClassifier: No seu config.yml, o threshold está definido como 0.6, o que significa que qualquer intenção com confiança abaixo de 60% será considerada como nlu_fallback.

    Ordem de Prioridade: As rules têm prioridade sobre as stories, e sua regra de fallback está capturando muitas mensagens que deveriam ser tratadas por outras intenções.

Soluções Recomendadas

    Ajuste o threshold do FallbackClassifier:
    yaml

- name: FallbackClassifier
  threshold: 0.3  # Reduza para 30% (valor mais comum)
  ambiguity_threshold: 0.1

Adicione um utter_default na regra de fallback:
Modifique sua rules.yml:
yaml

- rule: Fallback to web search
  steps:
    - intent: nlu_fallback
    - action: utter_default  # Adicione esta linha antes do web search
    - action: action_fallback_web_search

Melhore o pipeline de NLU:
Atualize seu config.yml com um pipeline mais robusto:
yaml

    pipeline:
      - name: WhitespaceTokenizer
      - name: RegexFeaturizer
      - name: LexicalSyntacticFeaturizer
      - name: CountVectorsFeaturizer
        analyzer: "char_wb"
        min_ngram: 1
        max_ngram: 4
      - name: DIETClassifier
        epochs: 100
        constrain_similarities: true
      - name: FallbackClassifier
        threshold: 0.3
        ambiguity_threshold: 0.1

    Adicione mais exemplos de treino:
    Seu nlu.yml poderia ter mais variações para cada intenção, especialmente para greet e goodbye.

    Verifique a ação de fallback:
    Certifique-se que action_fallback_web_search está retornando os resultados corretos e não está sendo acionada indevidamente.

Problemas Adicionais Identificados

    Intent ausente no domain:
    Você tem um warning sobre ask_uab_courses não estar definido no domain, mas no seu domain.yml eu vejo que está listado. Verifique se não há duplicações.

    Configuração de linguagem:
    Seu config.yml está em inglês (language: en), mas seu bot está em português. Mude para:
    yaml

    language: pt

    Adicione respostas padrão:
    No domain.yml, adicione respostas mais completas para cada intenção.

Recomendação Final

Depois de fazer essas alterações:

    Treine novamente o modelo:
    bash

rasa train

Teste com:
bash

rasa shell

Para depuração detalhada, use:
bash

    rasa shell --debug

Isso deve resolver o problema do fallback sendo acionado indevidamente. O bot deverá agora responder corretamente a saudações como "Olá" e "Boa tarde" sem tentar fazer pesquisa na web, a menos que realmente não entenda a mensagem.