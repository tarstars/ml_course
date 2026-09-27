previous lectures https://disk.yandex.ru/d/cweiLcy9ePRt7Q
ML 4 курс
https://education.yandex.ru/handbook/ml
vorontsov's course https://www.youtube.com/playlist?list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK



themes for test

* машинное обучение. примеры задач машинного обучения. классификация задач машинного обучения
* принцип наибольшего правдоподобия. для гауссово распределения ошибок вывод  метода наименьших квадратов из принципа наибольшего правдоподобия. вывод оценки параметра бернуллиевской случайной величины из метода наибольшего правдоподобия
* матрица объект-признак. задачи классификации и регрессии. функции потери для классификации и регрессии. отличие метрик от функции потерь
* работа с категориальными признаками. упорядоченные и неупорядоченные признаки. one-hot encoding. счётчики. hash trick. 
* линейная модель. l2 и l1 регуляризация. свойства этих видов регуляризации. может ли линейная модель уловить нелинейную зависимость? аналитическая формула для линейной регрессии; отступ
* метрики задачи бинарной классификации: precision, recall, tpr, fpr, ROC, ROC AUC, f1
* дерево: процедура разделения (split), энтропия и дисперсия, Gini impurity


Machine Learning: Examples of machine learning tasks. Classification of machine learning tasks.
Maximum Likelihood Principle: Deriving the least squares method from the maximum likelihood principle for Gaussian error distribution. Deriving the parameter estimate of a Bernoulli random variable using the maximum likelihood method.
Feature Matrix: Classification and regression tasks. Loss functions for classification and regression. Difference between metrics and loss functions.
Working with Categorical Features: Ordered and unordered features. One-hot encoding. Count encoding. Hash trick.
Linear Models: L2 and L1 regularization. Properties of these types of regularization. Can a linear model capture nonlinear dependencies? Analytical formula for linear regression; intercept.
Metrics for Binary Classification Tasks: Precision, recall, TPR (True Positive Rate), FPR (False Positive Rate), ROC, ROC AUC, F1-score.
Decision Trees: Splitting procedure, entropy and variance, Gini impurity.




* метрики бинарной классификации: precision, recall, f1, ROC, ROC AUC
* построение дерева для решения задач классификации и регрессии; функции потерь; жадный алгоритм построения сплита 
* бэггинг, random forest
* бустинг


Translation of the ML Lesson Program:

- Metrics of Binary Classification: Precision, Recall, F1 Score, ROC Curve, ROC AUC
- Decision Tree Construction for Classification and Regression Problems: Loss Functions; Greedy Algorithm for Split Construction
- Bagging and Random Forest
- Boosting





# метрики задачи бинарной классификации

* задачу кредитного скоринга; основые фичи в этой задаче; пример линейной модели, решающей эту задачу
* необходимость ввести порог для принятий решений
* метрики оценки качества классификатора с порогом: precision, recall, TPR, FPR
* ROC AUC как суммарная метрика оценки качества модели
* ROC как инструмент выбора порога
* f1 - среднее гармоническое precision, recall
* выбор порога как максимизация f1
* выбор порога проведением касательной к ROC

# построение дерева

* процедура split: выбираем значение в листе, минимизирующее функционал; выбираем сплит, минимизирующий функционал после уменьшения за счёт значения в листе
* вывод дисперсии и энтропии как MSE и logloss с подставленной эмпирической оценкой предсказания минимизирующего лосс по заданной выборке



Лекция
https://disk.yandex.ru/i/RegRiVkQ4pX27Q
Семинар
https://disk.yandex.ru/i/0kqAD4Oji--IaA


Темы с занятия 21 октября 2024

* алгоритм сплита при построении дерева для задач регрессии и классификации
* метрики качества сплита: дисперсия, энтропия, Gini impurity
* вывод метрик качества сплита из метрики решаемой задачи (MSE, logloss)
* определение энтропии, минимальное и максимальное значение энтропии случайной величины с k разными значениями
* бэггинг; Random Forest; самые важные гиперпараметры модели
* градиентно бустированные деревья: где здесь градиент?
* как выглядит оптимизация ROC с точки зрения построения леса решающих деревьев?



Вопросы на зачёт/экзамен:
* в чём "жадность" алгоритма построения сплита?
* вывод метрики logloss
* строгий вывод максимальной энтропии случайной величины с k разными значениями через метод множителей Лагранжа
* вывод формул для метрик сплита из метрик задачи
* Выражение и смысл Gini Impurity


Topics from the Lesson

The splitting algorithm when building a tree for regression and classification tasks
Split quality metrics: variance, entropy, Gini impurity
Deriving split quality metrics from the task's metrics (MSE, logloss)
Definition of entropy, minimum and maximum values of entropy for a random variable with k different values
Bagging; Random Forest; the most important hyperparameters of the model
Gradient-boosted trees: where is the gradient here?
How does ROC optimization look from the perspective of building a forest of decision trees?
Questions for Credit/Exam

What is the "greediness" of the splitting algorithm?
Derivation of the logloss metric
Rigorous derivation of the maximum entropy of a random variable with 
k different values using the method of Lagrange multipliers
Deriving formulas for split metrics from task metrics
Expression and meaning of Gini Impurity



К чему нужно быть готовым на промежуточной проверочной работе 31 октября:

Теория
* Машинное обучение. Примеры задач машинного обучения. Классификация задач машинного обучения
* Принцип наибольшего правдоподобия. Для гауссова распределения ошибок вывод  метода наименьших квадратов из принципа наибольшего правдоподобия. Вывод оценки параметра бернуллиевской случайной величины из метода наибольшего правдоподобия.
* Матрица объект-признак. Задачи классификации и регрессии. Функции потери для классификации и регрессии. Отличие метрик от функции потерь.
* Работа с категориальными признаками. Упорядоченные и неупорядоченные признаки. one-hot encoding. счётчики. hash trick.
* Линейная модель. l2 и l1 регуляризация. Свойства этих видов регуляризации. может ли линейная модель уловить нелинейную зависимость? аналитическая формула для линейной регрессии; отступ
* метрики задачи бинарной классификации: precision, recall, tpr, fpr, ROC, ROC AUC, f1
* дерево: процедура разделения (split), энтропия и дисперсия, Gini impurity
* задача кредитного скоринга; основые фичи в этой задаче; пример линейной модели, решающей эту задачу
* необходимость ввести порог для принятий решений
* метрики оценки качества классификатора с порогом: precision, recall, TPR, FPR
* ROC AUC как суммарная метрика оценки качества модели
* ROC как инструмент выбора порога
* f1 - среднее гармоническое precision, recall
* выбор порога как максимизация f1
* выбор порога проведением касательной к ROC
* построение дерева; процедура split: выбираем значение в листе, минимизирующее функционал; выбираем сплит, минимизирующий функционал после уменьшения за счёт значения в листе
* вывод дисперсии и энтропии как MSE и logloss с подставленной эмпирической оценкой предсказания минимизирующего лосс по заданной выборке
* метрики качества сплита: дисперсия, энтропия, Gini impurity
* вывод метрик качества сплита из метрики решаемой задачи (MSE, logloss)
* определение энтропии, минимальное и максимальное значение энтропии случайной величины с k разными значениями
* бэггинг; Random Forest; самые важные гиперпараметры модели
* градиентно бустированные деревья: где здесь градиент?
* как выглядит оптимизация ROC с точки зрения построения леса решающих деревьев?



Выкладки/вопросы:
* вывод МНК из метода наибольшего правдоподобия
* в чём "жадность" алгоритма построения сплита?
* вывод метрики logloss
* вывод оценки параметра бернуллиевской величины из минимизации logloss
* формула линейной регрессии
* формула для отступа (margin) логистической регрессии
* формула логистической регрессии
* аналитическое выражение для модели линейной регрессии с L2 регуляризацией
* строгий вывод максимальной энтропии случайной величины с k разными значениями через метод множителей Лагранжа
* вывод формул для метрик сплита из метрик задачи
* Выражение и смысл Gini Impurity
* выражения для precision, recall, f1. Процедура построения ROC и вычисления ROC AUC
* выражение для энтропии
* выражение для Gini impurity


What you need to be prepared for in the intermediate assessment on October 31:

Theory

Machine Learning: Examples of machine learning tasks. Classification of machine learning tasks.
Maximum Likelihood Principle: Derivation of the least squares method from the maximum likelihood principle for Gaussian error distribution. Derivation of the parameter estimate of a Bernoulli random variable using the maximum likelihood method.
Feature Matrix: Classification and regression tasks. Loss functions for classification and regression. The difference between metrics and loss functions.
Working with Categorical Features: Ordered and unordered features. One-hot encoding, counters, hash trick.
Linear Model: L2 and L1 regularization. Properties of these types of regularization. Can a linear model capture nonlinear dependencies? Analytical formula for linear regression; margin.
Metrics for Binary Classification Tasks: Precision, recall, TPR (True Positive Rate), FPR (False Positive Rate), ROC, ROC AUC, F1 score.
Decision Trees: Split procedure, entropy and variance, Gini impurity.
Credit Scoring Task: Key features in this task; example of a linear model solving this task.
Need to Introduce a Decision Threshold
Metrics for Evaluating Classifier Quality with a Threshold: Precision, recall, TPR, FPR.
ROC AUC as an Overall Metric for Model Quality Evaluation
ROC as a Tool for Threshold Selection
F1 Score: Harmonic mean of precision and recall.
Threshold Selection by Maximizing F1
Threshold Selection by Drawing a Tangent to the ROC Curve
Building a Decision Tree: Split procedure—choosing the value in the leaf that minimizes the functional; choosing the split that minimizes the functional after reduction due to the value in the leaf.
Derivation of Variance and Entropy as MSE and Log Loss: Using the empirical estimate of the prediction that minimizes loss on a given sample.
Metrics for Split Quality: Variance, entropy, Gini impurity.
Deriving Split Quality Metrics from Task Metrics: MSE, log loss.
Definition of Entropy: Minimum and maximum values of entropy for a random variable with 
𝑘
k different values.
Bagging and Random Forests: Most important hyperparameters of the model.
Gradient Boosted Trees: Where is the gradient here?
How ROC Optimization Appears in Building a Forest of Decision Trees
Calculations/Questions

Derivation of Ordinary Least Squares (OLS) from the Maximum Likelihood Method
What Makes the Split Algorithm "Greedy"?
Derivation of the Log Loss Metric
Derivation of the Parameter Estimate of a Bernoulli Variable by Minimizing Log Loss
Formula for Linear Regression
Formula for the Margin in Logistic Regression
Formula for Logistic Regression
Analytical Expression for a Linear Regression Model with L2 Regularization
Rigorous Derivation of the Maximum Entropy of a Random Variable with 
𝑘
k Different Values Using Lagrange Multipliers
Derivation of Formulas for Split Metrics from Task Metrics
Expression and Meaning of Gini Impurity
Expressions for Precision, Recall, F1 Score: Procedure for constructing ROC and calculating ROC AUC.
Expression for Entropy
Expression for Gini Impurity


Занятие 28 октября 2024


Теория
* SVD 
* формула, применение для предсказания активности в магазине; рекомендательной системе; сжатие картинки
* понятие факторизуемости, связь с правилами матричного умножения
* процедура выбора факторизуемых компонент, вносящих максимальный вклад

Выкладки/вопросы:
* написать выражения для матрицы при имеющемся SVD разложении на USV
* объяснить что такое факторизуемость, как она выводится из правил матричного умножения и как связана с выражением для матрицы при имеющихся матрицах USV
* свойства матриц U, S, V
* описание процедуры определения количества главных компонентов SVD разложения
* для матрицы, заданной выражением m_pq = exp(-(p-50)^2 - (q - 50)^2): объяснить, как матрица выглядит в виде серо-чёрной картинки, факторизуема ли матрица?


https://disk.yandex.ru/i/qjxtrhDmxXqNKA
https://disk.yandex.ru/i/K5Qgx0V_GrUASg


Занятие 4 ноября 2024

Теория 
* Доверительный интервалы и проверка гипотез
   * построение доверительного интервала нормального распределения
   * что такое: уровень доверия, мощность
   * уметь правильно проговаривать процедуру статистической проверки, что такое нулевая гипотеза, что такое статзначимость
  * ошибки первого и второго рода
* Эмбеддинги
  * что это такое?
  * word2vec модель
  * как обучается? (датасет, функция потерь, положительные и отрицательные примеры)
* Байес
  * Отличие классического подхода от Байесовского
  * оценка параметра распределения Гаусса и Бернулли в классическом и Байесовском случаях
  * параметр распределения и доверительный интервал в классическом и Байесовском подходе
  * понятие условной вероятности; формула Байеса; вывод формулы Байеса
  * наивный Байесовский классификатор; формулы
  * сопряженные распределения; какие распределения сопряжены с нормальным? с бернуллиевским?
  * получение оценки параметра гауссова распределения в классическом и в Байесовом подходах


Практика
* как строить доверительный интервал средствами sklearn? 
* решать задачки на формулу условной вероятности типа такой:
Ночью произошло ДТП с участием такси. В городе работают две компании такси — «Синие» и «Зелёные». «Зелёным» принадлежит 85% такси, «Синим» 15%. Свидетель аварии утверждает, что такси принадлежало «Синим». Следственный эксперимент показал, что ночью свидетель верно определяет цвет такси в 80%. Какова вероятность того, что такси действительно принадлежало «Синим»?

или такой 

Вирус XYZ вызывает серьёзное заболевание у одного человека из тысячи. Существует тест, показывающий, заражён человек XYZ или нет. Однако этот тест в 5% даёт ложное срабатывание — показывает факт заражения, когда на самом деле человек здоров. Предположим, что тестирование взятого наугад человека показало, что он XYZ-инфицирован. Какова вероятность того, что он на самом деле болен?



https://habr.com/ru/articles/855712/


https://disk.yandex.ru/d/cweiLcy9ePRt7Q



https://www.learndatasci.com/glossary/gini-impurity/


Программа занятия от 11 ноября 2024, будет добавлена к программе экзамена

Тема: "Введение в Deep Learning"

Общие вопросы
* обзор всех обсуждённых до сих пор методов, их связь с нейронками:
   * непараметрические методы: KNN, наивный баесовский классификатор
   * параметрические методы:
      * линейная модель: линейная регрессия, логистическая регрессия
      * SVM
      * деревья: линейность дерева
   * ансамбли: bagging (связь с bootstrap, bias-variance decomposition); boosting; stacking
* backpropagation: что такое, связь со свойствами дифференцирования
* матричное дифференцирование
* разные виды функций активации:
   * зачем в принципе нужна функция активации
   * на какие свойства нужно обращать внимание чтобы классифицировать функции активации?
   * примеры функций активации, объяснение их свойств
* теорема универсальной аппроксимации 

Практика:
* вывод формулы производной сигмоиды
* продифферинцировать log(det(A)) по матрице A

Лаба
girafe_ai/ml-course
week_03_fineart_classification
                 intro_to_pytorch
                 
                 
                 
 Программа занятия от 28 ноября 2024

Тема "Оптимизаторы глубокого обучения и регуляризации"

Из лекций и семинаров (https://disk.yandex.ru/d/cweiLcy9ePRt7Q) к этому занятию относятся "2023-11-30 Лекция - Радослав Нейчев - Методы регуляризации в нейронных сетях.mp4" и "2023-11-30 Семинар - Радослав Нейчев - Проблемы работы с нейронными сетями и их решения. Методы регуляризации в нейронных сетях.mp4"

Теоретические темы:
* Оптимизаторы. Основные идеи, в чём отличие. Adagard, Adadelta, RMSProp, Adam, ...
* Роль нормализации
* Batch normalization
* Виды регуляризации: L1, L2, dropout

In English:
* Optimizators. Main ideas, differences. Adagard, Adadelta, RMSProp, Adam, ...
* The Role of normalization
* Batch normalization
* Types of regularization: L1, L2, dropout



                
Семинар к прошедшему занятию:

"2023-11-30 Семинар - Радослав Нейчев - Проблемы работы с нейронными сетями и их решения. Методы регуляризации в нейронных сетях.mp4"

в семинаре разбираются темы:
* как в pytorch решить задачу многоклассовой классификации
* куча практических советов, как использовать разные фичи pytorch: Dataset, Compose, dropout
как правильно создавать и использовать device для работы с gpu
что такое эпоха
как влияет dropout

ссылка на тетрадку (https://github.com/girafe-ai/ml-course/tree/master/week0_08_dropout_batchnorm)



По поводу факторизуемости.

* я неправильно употребил термины. "факторизуемость" используется в физике, тем волновая функция факторизуется, если её можно представить как произведение двух функций, каждая из которых зависит только от пространства, или от времени
* я имел в виду "матрицы ранга 1": матрица ранга один как произведение двух векторов (https://math.stackexchange.com/questions/1545118/a-rank-one-matrix-is-the-product-of-two-vectors) 
* и, соответственно, теорема: любую матрицу можно представить как сумму матриц ранга 1 (https://math.stackexchange.com/questions/1213411/how-to-represent-matrix-as-the-sum-of-rank-one-matrices) 

На эту тему есть много разных ресурсов, например:
* связь нормы матрицы и разложение по одноранговым матрицам (https://websites.uwlax.edu/twill/svd/norm/index.html) 
* матричная факторизация (https://education.yandex.ru/handbook/ml/article/matrichnaya-faktorizaciya) в хэндбуке ШАД


Темы занятия 25 ноября 2024

* RNN
  * что такое токен
  * что такое эмбеддинг
  * обучаемый эмбеддинг = OHE + матрица
  * состояние RNN как фича-вектор фиксированной длины из контекста произвольной длины
  * устройство RNN: формула
  * обучение RNN: целевая функция потерь, что обучаем
  * инициализация начального состояния RNN
  * статья Карпаты про RNN (https://karpathy.github.io/2015/05/21/rnn-effectiveness/) 
  * код обучения побуквенной RNN (https://gist.github.com/karpathy/d4dee566867f8291f086) 
  * LSTM - посмотреть в лекции
     * чем LSTM отличается от RNN ?
     * какая идея?
     * какие преимущества?
     
     
     
   https://t.me/ai_machinelearning_big_data/6206
   
   
   Тетрадка (https://github.com/tarstars/dl/blob/main/rnn/generate_text_00.ipynb) в которой готовится датасет для генерации названий географических объектов с помощью RNN. Завтра продолжаем с этой точки.
   
   
  Матричное дифференцирование

😱 В ML периодически возникает потребность взять производную от матричного выражения, как правило имеющую вид «производная вектора по вектору». Если вас всегда ломало разобраться с матричным дифференцированием, а строгое формальное изложение матричных производных через дифференциалы вводило в уныние, я недавно записал видео с простым объяснением, как это работает. (https://youtu.be/aAAcBWyS_YI?si=3nbu29cu3M13a26J)

🎭 В ролике есть определенная драматургия: будет момент, где мы с вами сами начнем придумывать матричную производную так, чтобы она в частном случае превращалась в уже привычный нам градиент. И оттуда станет ясно, почему матричная производная именно такая, а не, например, транспонированная. 

🤓 Всем, кто хочет шарить за ML чуть глубже, чем «ну я тут что-то обучил и в докер завернул, а как работает не мое дело» - рекомендую к просмотру. Жить без матричных производных можно, но компактно оперировать формулами очень удобно.


Новые вопросы.

Теория
* Что такое RNN?
* Из каких блоков состоит RNN?
* Формулы в RNN
* Какую задачу позволяет решить? Как с помощью RNN решать задачу определения тональности текста?
* Как происходить обучение RNN?
* Как должен выглядеть датасет для RNN?
* К какой категории задач машинного обучения относится обучение RNN?
* Формула для функции потерь RNN
* Из каких матриц состоит RNN? Какое назначение у этих матриц? Какие у них размеры?
* Что происходит в прямом и обратном проходе RNN?
* Что нужно сделать чтобы сгенерировать текст с помощью RNN? Что такое температура генерации?
* Как реализовывается Adagard на numpy?
* Какую роль играет экспоненциальное сглаживание в программе, которую мы разобрали? Формула экспоненциального сглаживания. Зачем оно нужно в этой задаче?
* Какой смысл символов < и > в этой задаче? Для чего эти символы использовались?

Выкладки
* Найти производную по аргументу от tahn
* производная функции потерь RNN по логитам
* шаги бэктрекинга
   * слагаемое вектора
   * производная по элементам матрицы
   * производная вектора умноженного на матрицу
 
Практика программирования
* как прочитать формат файла, где в каждой строке хранится json ?
* выделить содержимое заданных полей
* создать уникальный список слов
* полный алфавит используемый в списке слов
* построить прямой проход RNN
* обратный проход с backpropagation; как реализуется на numpy



The same program in English:

New Questions

### Theory
- What is an RNN (Recurrent Neural Network)?
- What components make up an RNN?
- Formulas in RNN.
- What problem can RNN solve? How can RNN be used to solve the problem of text sentiment analysis?
- How is RNN trained?
- What should a dataset for RNN look like?
- What category of machine learning tasks does RNN training fall under?
- Formula for the loss function in RNN.
- What matrices make up an RNN? What are their purposes? What are their dimensions?
- What happens during the forward and backward passes in RNN?
- What needs to be done to generate text using RNN? What is generation temperature?
- How is Adagrad implemented in numpy?
- What role does exponential smoothing play in the program we reviewed? Provide the formula for exponential smoothing. Why is it needed for this task?
- What is the significance of the symbols < and > in this task? What were these symbols used for?

### Derivations
- Find the derivative of the argument for tanh.
- Derivative of the RNN loss function with respect to logits.
- Steps for backtracking:
  - Vector summand.
  - Derivative with respect to matrix elements.
  - Derivative of a vector multiplied by a matrix.

### Programming Practice
- How to read a file format where each line contains JSON?
- Extract the contents of specified fields.
- Create a unique list of words.
- Create the full alphabet used in the word list.
- Implement the forward pass of RNN.
- Implement the backward pass with backpropagation; how is it implemented in numpy?



Домашка по генерации стихов

Ноутбук: https://github.com/girafe-ai/ml-course/blob/23s_dd_ml/homeworks/hw09_poetry/assignment_poetry.ipynb
Контест: https://contest.yandex.ru/contest/73093/enter




 
Всем привет!

Поздравляю со скорым началом весеннего семестра.

На занятии 3 февраля будем разбирать материалы 

лекции (https://disk.yandex.ru/i/KRHqfKpbCXvmww) 
и 
семинара (https://disk.yandex.ru/i/fDTjaONcGQ5BUQ) 

Наша задача в одну пару втиснуть материал двух пар, поэтому предлагаю видосики посмотреть и подумать над ними, чтобы мы не целину пропахивали, а возделывали уже подготовленную почву.



    


Был вопрос, почему предварительная установка эквивалентнта регуляризации. Пара документов почитать:
* wiki (https://en.wikipedia.org/wiki/Early_stopping) 
* medium (https://medium.com/@rahuljain13101999/why-early-stopping-works-as-regularization-b9f0a6c2772)



документ (https://github.com/tarstars/dl/blob/main/spring_2025_ru.md) по мотивам вчерашнего занятия




Домашнее задание по терминам: написать своим языком короткие но содержательные объяснения. В чём заключается трюк, как его реализовать, для решения какой проблемы используется. Список терминов в файле (https://github.com/tarstars/dl/blob/main/spring_2025_ru.md) . Блок - "З февраля 2025". Эти трюки разобраны в лекции ШАД, на которую дана ссылка в сообщении от 28 Января.



запись (https://disk.yandex.ru/i/futDP5_fqSr8oA) семинара 2025-02-14



https://disk.yandex.ru/i/WQ6ceivbCDsK_Q


Запись семинара по трансформерам: https://disk.yandex.ru/i/L0px5-2cf_nkAQ


Лекции Карпаты (https://www.youtube.com/c/AndrejKarpathy) 
Если хочется хорошо разобраться, как устроен трансформер и как обучаются большие языковые модели


обучение с подкреплением лекция и семинар (https://disk.yandex.ru/i/jnjkHbg3pyTjoA) 

интерпретация нейросетей лекция (https://disk.yandex.ru/i/uJMRUWTJ8DAddQ)
интерпретация нейросетей семинар (https://disk.yandex.ru/i/Qlr4aRp86burzA)


Репозитарий (https://github.com/yandexdataschool/Practical_RL/tree/master/week01_intro) к семинару по RL


Тетрадка для семинара по RL

https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/seminar_gymnasium_interface.ipynb



Ссылки на записи, которые обещал.


интерпретация нейросетей, лекция (https://disk.yandex.ru/i/uJMRUWTJ8DAddQ) 
интерпретация нейросетей, семинар (https://disk.yandex.ru/i/Qlr4aRp86burzA) 
инструменты разработки (https://disk.yandex.ru/i/VqhSElsdrdnkhA)




✔️ Transformers & LLMs cheatsheet for Stanford's CME 295

Это репозиторий с набором визуальных шпаргалок, посвященных трансформерам и большим языковым моделям (LLM).

🌟  Материалы основаны на курсе CME 295 "Трансформеры и большие языковые модели", читаемом в Стэнфордском университете.

Это не книга или подробный туториал, а именно сжатые, наглядные обзоры ключевых концепций. Используются диаграммы, краткие пояснения, основные формулы и определения.

✔️Содержание: Охватывает фундаментальные темы, необходимые для понимания LLM:
✔️Механизм внимания (Attention Mechanism)
✔️Архитектура Трансформера (Transformer Architecture)
✔️Позиционное кодирование (Positional Encoding)
✔️Предобучение (Pre-training) и его цели (например, Masked Language Modeling, Next Sentence Prediction)
✔️Дообучение (Fine-tuning) для конкретных задач
✔️Промптинг (Prompting) и различные его техники
✔️Оценка (Evaluation) LLM
✔️Другие важные концепции в этой области.

Идеально подходит для повторения материала или для того, чтобы быстро схватить суть той или иной концепции.

🟡Github (https://github.com/afshinea/stanford-cme-295-transformers-large-language-models)
🟡Multi-Head Attention Shape Transformations  (https://github.com/rasbt/LLMs-from-scratch/discussions/595)

#ml #cheatsheets #transformers













