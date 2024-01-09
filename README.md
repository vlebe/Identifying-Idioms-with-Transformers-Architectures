# PSTALN

## Baseline 

Notre baseline est une simple mémorisation des expressions du texte qui lui est donné en entraînement. Le principe est le suivant : 
- Pour un texte en entrée du modèle baseline on construit une liste d'expressions, où chaque expression est une liste de mots labellisés comme expression avec au maximum 1 mot non labéllisé comme expression entre chaque mot faisant partie de l'expression, par exemple :
    - **avoir** pour **but** (les mots en **gras** sont labéllisés comme expression) donnera l'expression suivante : ['avoir', 'but']
    - **avoir** pour principal **but** ne sera pas retenu comme une expression

Ainsi pour calculer le score de notre modèle nous récupérons les listes obtenues avec cette méthode pour le corpus train et le corpus test. Puis nous regardons combien des expressions obenues pour test étaient dans la liste d'expressions du train. 
- Avec cette première approche nous obtenons un score de 0.15.

Comme cela nous paraissait faible nous avons pris la décision de plutôt stocker le lemme de chaque mot dans les expressions, par exemple : 
- *va en guerre* devient *aller en guerre* ou encore *il faut tenir compte* devient *il falloir tenir compte*, etc...
- De cette manière avec la même méthodologie d'évaluation nous obtenons un score de 0.26 qui sera notre référence pour ce projet.  