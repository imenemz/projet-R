# Payment Default Prediction
# Projet d'analyse - Informatique décisionnelle

# Chargement des données du projet
# le setwd permet de se placer dans le dossier du projet
setwd("C:/Users/ASUS/OneDrive - Université Côte d'Azur/imene_uni/L3/S5/info_de/projet")

# Nettoyage de l'environnement
rm(list = ls())
set.seed(123)

# df contient les données avec le défaut connu
# df_new contient les clients a prédire

df <- read.csv("Data Projet.csv", stringsAsFactors = FALSE)
df_new <- read.csv("Data Projet New.csv", stringsAsFactors = FALSE)

# Transformation de la variable cible en facteur

df$defaut <- factor(df$defaut, levels = c("Non", "Oui"))

# Gestion des valeurs manquantes
df$age[df$age == 999] <- NA
df$adresse[df$adresse == 999] <- NA

# on remplace les NA par la médiane
# la médiane est robuste aux valeurs extremes
df$age[is.na(df$age)] <- median(df$age, na.rm = TRUE)
df$adresse[is.na(df$adresse)] <- median(df$adresse, na.rm = TRUE)

# Transformation des variables catégorielles
# education est ordonnée car les niveaux ont un sens
df$education <- factor(
  df$education,
  levels = c("Niveau bac", "Bac+2", "Bac+3", "Bac+4", "Bac+5 et plus"),
  ordered = TRUE
)

# categorie est une variable qualitative simple
df$categorie <- factor(df$categorie)

# Séparation apprentissage / test

index <- unlist(
  tapply(seq_len(nrow(df)), df$defaut,
         function(x) sample(x, floor(0.7 * length(x))))
)

train <- df[index, ]
test  <- df[-index, ]

# Suppression des niveaux inutilisés
# permet d'éviter des problemes dans les modèles
train <- droplevels(train)
test  <- droplevels(test)

# Suppression des variables factorielles avec un seul niveau
# ces variables ne sont pas utiles pour la prédiction
one_level_factors <- names(train)[
  sapply(train, function(x) is.factor(x) && length(unique(x)) < 2)
]

train <- train[, !(names(train) %in% one_level_factors)]
test  <- test[, !(names(test) %in% one_level_factors)]

# Apprentissage du modèle de régression logistique
# ce modèle est adapté a une variable cible binaire
model_log <- glm(defaut ~ ., data = train, family = binomial)

# Prédiction des probabilités sur le jeu de test
proba_log <- predict(model_log, test, type = "response")

# Seuil classique a 0.5
pred_log_05 <- factor(ifelse(proba_log >= 0.5, "Oui", "Non"),
                      levels = c("Non", "Oui"))

# Matrice de confusion et rappel pour le seuil 0.5
conf_log_05 <- table(pred_log_05, test$defaut)
recall_log_05 <- conf_log_05["Oui","Oui"] /
  (conf_log_05["Oui","Oui"] + conf_log_05["Non","Oui"])

# Seuil abaissé a 0.4
# permet de mieux detecter les clients a risque
pred_log_04 <- factor(ifelse(proba_log >= 0.4, "Oui", "Non"),
                      levels = c("Non", "Oui"))

# Matrice de confusion et rappel pour le seuil 0.4
conf_log_04 <- table(pred_log_04, test$defaut)
recall_log_04 <- conf_log_04["Oui","Oui"] /
  (conf_log_04["Oui","Oui"] + conf_log_04["Non","Oui"])

# Calcul de la courbe ROC et de l'AUC pour la régression logistique
library(pROC)

roc_log <- roc(test$defaut, proba_log, levels = c("Non","Oui"))
auc_log <- auc(roc_log)

# Apprentissage d'un arbre de décision
# ce modèle est plus simple mais souvent moins performant
library(rpart)

model_tree <- rpart(defaut ~ ., data = train, method = "class")

# Prédiction des probabilités avec l'arbre
proba_tree <- predict(model_tree, test, type = "prob")[,"Oui"]

pred_tree <- factor(ifelse(proba_tree >= 0.5, "Oui", "Non"),
                    levels = c("Non","Oui"))

# Matrice de confusion et rappel pour l'arbre
conf_tree <- table(pred_tree, test$defaut)
recall_tree <- conf_tree["Oui","Oui"] /
  (conf_tree["Oui","Oui"] + conf_tree["Non","Oui"])

# Courbe ROC et AUC pour l'arbre de décision
roc_tree <- roc(test$defaut, proba_tree, levels = c("Non","Oui"))
auc_tree <- auc(roc_tree)

# Affichage de la courbe ROC pour comparer les deux modèles
dev.new()

plot(roc_log, col = "lightblue", lwd = 2,
     main = "ROC Curve: Logistic vs Decision Tree")
lines(roc_tree, col = "salmon", lwd = 2)
abline(a = 0, b = 1, lty = 2)

legend("bottomright",
       legend = c(
         paste("Logistic (AUC =", round(auc_log,3), ")"),
         paste("Decision Tree (AUC =", round(auc_tree,3), ")")
       ),
       col = c("lightblue", "salmon"),
       lwd = 2)

# Analyse exploratoire : relation entre categorie et défaut
dev.new()
barplot(
  table(df$categorie, df$defaut),
  beside = TRUE,
  col = c("lightgreen", "pink"),
  legend = TRUE,
  main = "Categorie vs Default",
  ylab = "Number of clients"
)

# Visualisation de la matrice de confusion (seuil 0.4)
dev.new()
conf_mat <- table(pred_log_04, test$defaut)

image(conf_mat,
      col = heat.colors(256),
      main = "Confusion Matrix (Logistic, threshold = 0.4)",
      xlab = "Actual",
      ylab = "Predicted")

text(expand.grid(1:2, 1:2),
     labels = as.vector(conf_mat),
     col = "black")

# Distribution des probabilités prédites avant le choix du seuil
dev.new()
plot(density(proba_log[test$defaut == "Non"]),
     col = "red", lwd = 2,
     main = "Predicted Probabilities (Logistic)",
     xlab = "Predicted probability")
lines(density(proba_log[test$defaut == "Oui"]),
      col = "blue", lwd = 2)

legend("topright",
       legend = c("Non default", "Default"),
       col = c("red", "blue"),
       lwd = 2)

# Comparaison finale des modèles
comparison <- data.frame(
  Model = c("Logistic (0.5)", "Logistic (0.4)", "Decision Tree"),
  Recall = c(recall_log_05, recall_log_04, recall_tree),
  AUC = c(auc_log, auc_log, auc_tree)
)

print(comparison)

# Application du modèle final aux nouvelles données
# on utilise le seuil 0.4 retenu precedemment

df_new$age[df_new$age == 999] <- NA
df_new$adresse[df_new$adresse == 999] <- NA

df_new$age[is.na(df_new$age)] <- median(df$age, na.rm = TRUE)
df_new$adresse[is.na(df_new$adresse)] <- median(df$adresse, na.rm = TRUE)

df_new$education <- factor(df_new$education,
                           levels = levels(df$education),
                           ordered = TRUE)

df_new$categorie <- factor(df_new$categorie,
                           levels = levels(df$categorie))

# Prédiction sur les nouveaux clients
proba_new <- predict(model_log, df_new, type = "response")
pred_new <- ifelse(proba_new >= 0.4, "Oui", "Non")

# Création du fichier final a déposer sur Moodle
results <- data.frame(
  client = df_new$client,
  prediction = pred_new,
  probability = proba_new
)

write.csv(results, "FINAL_RESULTS.csv", row.names = FALSE)
