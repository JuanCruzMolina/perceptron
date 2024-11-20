// Definición de pines
const int botonPin = 2;      // Pin al que está conectado el botón
const int ledPin = 13;       // Pin al que está conectado el LED digital
const int potPin = A0;       // Pin al que está conectado el potenciómetro
const int ledPWM = 9;        // Pin PWM para el LED analógico

// Variables para almacenar el estado del botón y la lectura del potenciómetro
int estadoBoton = 0;
int lecturaPot = 0;

void setup() {
  // Inicializar el pin del LED como salida
  pinMode(ledPin, OUTPUT);
  
  // Inicializar el pin del botón como entrada
  pinMode(botonPin, INPUT_PULLUP); // Usa resistor interno de pull-up
  
  // Inicializar el pin PWM como salida
  pinMode(ledPWM, OUTPUT);
  
  // Iniciar la comunicación serial para depuración
  Serial.begin(9600);
}

void loop() {
  // Leer el estado del botón
  estadoBoton = digitalRead(botonPin);
  
  // Controlar el LED digital basado en el botón
  if (estadoBoton == LOW) { // LOW significa que el botón está presionado
    digitalWrite(ledPin, HIGH); // Enciende el LED
  } else {
    digitalWrite(ledPin, LOW);  // Apaga el LED
  }
  
  // Leer el valor del potenciómetro (0-1023)
  lecturaPot = analogRead(potPin);
  
  // Mapear el valor leído (0-1023) a un valor PWM (0-255)
  int valorPWM = map(lecturaPot, 0, 1023, 0, 255);
  
  // Aplicar el valor PWM al LED analógico
  analogWrite(ledPWM, valorPWM);
  
  // Enviar datos al monitor serial para depuración
  Serial.print("Botón: ");
  Serial.print(estadoBoton == LOW ? "Presionado" : "Liberado");
  Serial.print(" | Potenciómetro: ");
  Serial.print(lecturaPot);
  Serial.print(" | Valor PWM: ");
  Serial.println(valorPWM);
  
  // Esperar 100 ms antes de la siguiente lectura
  delay(100);
}
