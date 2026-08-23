// Mecanum Wheel Robot Car - ESP32-S3 (DevKitC-1) version
// Basado en https://robotlk.com/web-controlled-mecanum-wheel-robot-car-using-esp32/
// Cambios respecto al original:
//   1. Pines remapeados: el original usa GPIO 26-33, reservados/no expuestos en ESP32-S3.
//   2. Corregidas forwardLeft/forwardRight/backLeft/backRight (estaban cruzadas izq/der).
//   3. Watchdog: si no llega ningun comando en 500ms, detiene motores (evita carrito descontrolado
//      si se pierde WiFi o no llega el evento touchend/mouseup).

#include <WiFi.h>
#include <WebServer.h>
#include "credentials.h" // define WIFI_SSID y WIFI_PASSWORD (no se sube a git, ver README)

// Se conecta a tu red de casa (modo estación) en vez de crear su propio AP.
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

WebServer server(80);

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

  // Baja un poco la potencia de TX: reduce el pico de corriente al conectar
  // (mitiga brownouts, pero el arreglo real es de alimentación, ver notas).
  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  WiFi.begin(ssid, password);

  Serial.print("Conectando a ");
  Serial.println(ssid);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(300);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("Conectado. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("No se pudo conectar al WiFi. Revisa SSID/password.");
  }

  server.on("/", handleRoot);
  server.on("/cmd", handleCommand);
  server.begin();

  lastCmdTime = millis();
}

void loop() {
  server.handleClient();

  // Watchdog de seguridad: si no llega comando reciente, detener motores
  if (millis() - lastCmdTime > CMD_TIMEOUT) {
    stopCar();
  }
}

// ===== HTML PAGE =====
void handleRoot() {
  String html = R"rawliteral(
  <html>
  <head>
  <title>Mecanum Car</title>
  <style>
    button {width:100px;height:60px;font-size:16px;margin:5px;}
  </style>
  <script>
    function sendCmd(cmd){fetch(`/cmd?move=${cmd}`);}
    function hold(btn,cmd){
      btn.addEventListener('mousedown',()=>sendCmd(cmd));
      btn.addEventListener('mouseup',()=>sendCmd('S'));
      btn.addEventListener('touchstart',(e)=>{e.preventDefault();sendCmd(cmd);});
      btn.addEventListener('touchend',()=>sendCmd('S'));
      btn.addEventListener('touchcancel',()=>sendCmd('S'));
    }
    window.onload=()=>{
      ['F','B','SL','SR','RL','RR','FL','FR','BL','BR'].forEach(id=>{
        hold(document.getElementById(id),id);
      });
    }
  </script>
  </head>
  <body style='text-align:center;'>
    <h2>Mecanum Wheel Robot Car</h2>
    <div>
      <button id='FL'>F-L</button>
      <button id='F'>F</button>
      <button id='FR'>F-R</button>
    </div>
    <div>
      <button id='SL'>S-L</button>
      <button id='SR'>S-R</button>
    </div>
    <div>
      <button id='BL'>B-L</button>
      <button id='B'>B</button>
      <button id='BR'>B-R</button>
    </div>
    <div>
      <button id='RL'>R-L</button>
      <button id='RR'>R-R</button>
    </div>
    <br>
    <button onclick="sendCmd('S')">STOP</button>
  </body></html>
  )rawliteral";
  server.send(200, "text/html", html);
}

// ===== Handle Commands =====
void handleCommand() {
  lastCmdTime = millis();

  String move = server.arg("move");
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
  else stopCar();

  server.send(200, "text/plain", "OK");
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
