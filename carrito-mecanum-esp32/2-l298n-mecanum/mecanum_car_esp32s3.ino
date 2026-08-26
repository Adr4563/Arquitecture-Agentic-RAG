// Mecanum Wheel Robot Car - ESP32-S3 (DevKitC-1) version
// Basado en https://robotlk.com/web-controlled-mecanum-wheel-robot-car-using-esp32/
// Cambios respecto al original:
//   1. Pines remapeados: el original usa GPIO 26-33, reservados/no expuestos en ESP32-S3.
//   2. Corregidas forwardLeft/forwardRight/backLeft/backRight (estaban cruzadas izq/der).
//   3. Watchdog: si no llega ningun comando en 500ms, detiene motores (evita carrito descontrolado
//      si se corta el cable o el proceso del otro lado se cuelga).
//   4. Control por Serial (USB) en vez de WiFi/HTTP: el ESP32 va conectado por cable directo
//      a la Raspberry Pi, así que no hace falta radio de por medio — un comando por línea,
//      mismo protocolo de texto (F/B/SL/SR/RL/RR/FL/FR/BL/BR/S) que antes viajaba como
//      ?move=<código> por HTTP. Ver Clients/Carrito_Client.py en deploy-raspberry-standalone/.

// Motor pins (ESP32-S3-DevKitC-1: evita 0,3,19,20,26-32,43-46)
int IN1 = 4,  IN2 = 5;   // Front Left
int IN3 = 6,  IN4 = 7;   // Front Right
int IN5 = 15, IN6 = 16;  // Back Left
int IN7 = 17, IN8 = 18;  // Back Right

unsigned long lastCmdTime = 0;
const unsigned long CMD_TIMEOUT = 500; // ms

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(IN5, OUTPUT); pinMode(IN6, OUTPUT);
  pinMode(IN7, OUTPUT); pinMode(IN8, OUTPUT);

  stopCar();
  lastCmdTime = millis();

  Serial.println("Listo. Comandos por Serial: F/B/SL/SR/RL/RR/FL/FR/BL/BR/S, uno por línea.");
}

void loop() {
  // Un comando por línea (terminada en '\n'), igual que antes era un query
  // param por request — se lee de a línea completa, no byte a byte, para no
  // ejecutar un comando a medio escribir.
  if (Serial.available()) {
    String linea = Serial.readStringUntil('\n');
    linea.trim();
    if (linea.length() > 0) {
      ejecutarComando(linea);
      lastCmdTime = millis();
    }
  }

  // Watchdog de seguridad: si no llega comando reciente, detener motores
  // (mismo timeout que antes; ahora protege contra un cable desconectado o
  // el proceso Python del otro lado colgado, no contra WiFi caído).
  if (millis() - lastCmdTime > CMD_TIMEOUT) {
    stopCar();
  }
}

// ===== Interpretar comando =====
void ejecutarComando(const String& move) {
  if (move == "F") forward();
  else if (move == "B") backward();
  else if (move == "SL") strafeLeft();
  else if (move == "SR") strafeRight();
  else if (move == "RL") rotateLeft();
  else if (move == "RR") rotateRight();
  else if (move == "FL") forwardLeft();
  else if (move == "FR") forwardRight();
  else if (move == "BL") backLeft();
  else if (move == "BR") backRight();
  else stopCar(); // incluye "S" y cualquier texto no reconocido
}

// ===== Motor Control Logic =====
void stopCar(){
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
  digitalWrite(IN5,LOW); digitalWrite(IN6,LOW);
  digitalWrite(IN7,LOW); digitalWrite(IN8,LOW);
}

void forward(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);
}

void backward(){
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH);
  digitalWrite(IN5,LOW); digitalWrite(IN6,HIGH);
  digitalWrite(IN7,LOW); digitalWrite(IN8,HIGH);
}

void strafeRight(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  // FL
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); // FR
  digitalWrite(IN5,LOW);  digitalWrite(IN6,HIGH); // BL
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);  // BR
}

void strafeLeft(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);
  digitalWrite(IN7,LOW);  digitalWrite(IN8,HIGH);
}

void rotateRight(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH);
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);
  digitalWrite(IN7,LOW);  digitalWrite(IN8,HIGH);
}

void rotateLeft(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  digitalWrite(IN5,LOW);  digitalWrite(IN6,HIGH);
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);
}

// --- Diagonales corregidas (antes estaban cruzadas izq/der) ---

void forwardLeft(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  // FL forward
  digitalWrite(IN3,LOW);  digitalWrite(IN4,LOW);  // FR stop
  digitalWrite(IN5,LOW);  digitalWrite(IN6,LOW);  // BL stop
  digitalWrite(IN7,HIGH); digitalWrite(IN8,LOW);  // BR forward
}

void forwardRight(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,LOW);  // FL stop
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  // FR forward
  digitalWrite(IN5,HIGH); digitalWrite(IN6,LOW);  // BL forward
  digitalWrite(IN7,LOW);  digitalWrite(IN8,LOW);  // BR stop
}

void backLeft(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,LOW);  // FL stop
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); // FR backward
  digitalWrite(IN5,LOW);  digitalWrite(IN6,HIGH); // BL backward
  digitalWrite(IN7,LOW);  digitalWrite(IN8,LOW);  // BR stop
}

void backRight(){
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); // FL backward
  digitalWrite(IN3,LOW);  digitalWrite(IN4,LOW);  // FR stop
  digitalWrite(IN5,LOW);  digitalWrite(IN6,LOW);  // BL stop
  digitalWrite(IN7,LOW);  digitalWrite(IN8,HIGH); // BR backward
}
