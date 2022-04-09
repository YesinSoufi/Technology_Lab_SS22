# <h1>Technology_Lab_SS22</h1>

<p>Elena Müller 42616</p>
<p>Osman Kaplan 42581 </p>
<p>Niklas Öxle 42614 </p>
<p>Sascha Lehmann 42599 </p>
<p>Yesin Soufi 42612 </p>
<p>Jakob Schaal 42613 </p>

Aufgabenverteilung: [Kanban-Board zum Projekt](https://github.com/YesinSoufi/Technology_Lab_SS22/projects/2)

<h2>Projektbeschreibung</h2>

Im Rahmen des Projekts Technology Lab wird ein System entwickelt, das mithilfe eines neuronalen Netzwerkes anhand unbekannter Samples einen Musikstream generiert . Das Projektvorgehen ist in drei Schritte gegliedert:

1. Maschinelles Lernen von Sequenzen
    * Herstellung von Trainingsdaten durch Zerlegung von Musikstücken in Samples
    * Schaffen einer geeigneten Infrastruktur
2. Rekonstruktion des ursprünglichen Tracks bzw. Bilden von Sequenzen adaptierbarer Musik auf Basis bekannter Samples
    * Zusammensetzen (Resampling) der zuvor zerstückelten Musik
3. Rekombination von Musikstreams bzw. Lernen von Sequenzen von adaptierbarer Musik auf Basis zuvor unbekannter Samples
    * Resampling von Samples, die dem Neuronalen Netz zuvor nicht zugeführt  wurden
Im Fokus des Projekts steht neben Entwicklung eines lauffähigen Prototyps v.a. der wissenschaftliche Erkenntnisgewinn.

<h3>Forschungsfragen:</h3> 
Ziel des Projekts ist die Beantwortung der folglich definierten Forschungsfragen. 

   * Lässt sich ein „hörbarer“ (d.h. kaum von einem manuell gemixten Set unterscheidbarer) Stream durch das Zerschneiden und anschließende Resampling von Tracks auf Basis des Maschinellen Lernens erzeugen? 
   * Eignen sich gleichlange Samples oder stochastisch gewählte Samples besser als Datengrundlage? Wie lange ist eine optimale Sampledauer?
   * Wie ähnlich müssen sich die Ursprungstracks bzgl. Stimmung, Genre etc. sein? 
   * Müssen Metadaten wie Lautstärke, Tempo, Spektrum, etc. mit einbezogen werden?

<h4>Systemanforderungen:</h4> 
Zusätzlich zu den vorgegebenen Projektschritten (Challenges) und den sich daraus ableitenden Systemanforderungen wurden Personas und User Stories erarbeitet. 

  * [Personas](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/Personas.md)
  * [User Stories](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/UserStories.md)
 

<h4>Systementwurf:</h4> 

  * Scenarios and Glossar
  * Components diagram
  * [Machine learning Konzept](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/assets/MachineLearningKonzept.pdf)
  * [Training data](https://github.com/YesinSoufi/Technology_Lab_SS22/blob/main/TrainingData.md)