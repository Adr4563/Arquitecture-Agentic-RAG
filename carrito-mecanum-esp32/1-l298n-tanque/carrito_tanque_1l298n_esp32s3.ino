// Carrito Mecanum en modo TANQUE - ESP32-S3 (DevKitC-1) - variante con 1 solo L298N
// Basado en ../2-l298n-mecanum/mecanum_car_esp32s3.ino, adaptado para quien solo tiene 1 L298N
// (4 pines IN1-IN4, 2 canales) en vez de 2 L298N (8 pines IN1-IN8, 4 canales).
//
// Diferencia clave: con solo 2 canales de control, las 4 ruedas mecanum quedan
// agrupadas de a pares por LADO del chasis (los 2 motores izquierdos -FL+BL- en
// paralelo al mismo canal, los 2 derechos -FR+BR- en paralelo al otro canal) en
// vez de controlarse las 4 de forma independiente. Eso significa que se pierde
// el desplazamiento lateral (strafe) y las 4 diagonales: solo quedan adelante,
// atras, y giro en el sitio tipo tanque (un lado adelante + el otro atras).
// Ver README.md de esta carpeta y docs/cableado-1l298n-tanque.html para el detalle
// de cableado (motores en paralelo por canal) y sus implicancias de corriente.

#include <WiFi.h>
#include <WebServer.h>
#include "credentials.h" // define WIFI_SSID y WIFI_PASSWORD (no se sube a git, ver README)

// Se conecta a tu red de casa (modo estación) en vez de crear su propio AP.
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

WebServer server(80);

// Motor pins (ESP32-S3-DevKitC-1: evita 0,3,19,20,26-32,43-46)
// Un solo L298N: 2 canales, cada uno mueve 2 motores en paralelo (mismo lado del chasis).
int IN1 = 4, IN2 = 5;   // Canal A -> lado IZQUIERDO (FL + BL en paralelo)
int IN3 = 6, IN4 = 7;   // Canal B -> lado DERECHO   (FR + BR en paralelo)

unsigned long lastCmdTime = 0;
const unsigned long CMD_TIMEOUT = 500; // ms

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

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
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>Carrito Tanque (1 L298N)</title>
  <style>
    body{font-family:sans-serif; text-align:center; margin-top:20px;}
    button{width:80px; height:80px; font-size:28px; margin:4px;}
    button.rotate{width:100px; font-size:20px;}
    #stop{width:170px; height:50px; font-size:18px; margin-top:15px; color:red;}
    p.note{color:#888; font-size:13px; max-width:280px; margin:20px auto 0;}
  </style>
  <script>
    function sendCmd(cmd){ fetch(`/cmd?move=${cmd}`); }
    function hold(btn,cmd){
      btn.addEventListener('mousedown',()=>sendCmd(cmd));
      btn.addEventListener('mouseup',()=>sendCmd('S'));
      btn.addEventListener('touchstart',(e)=>{e.preventDefault(); sendCmd(cmd);});
      btn.addEventListener('touchend',()=>sendCmd('S'));
      btn.addEventListener('touchcancel',()=>sendCmd('S'));
    }
    window.onload=()=>{
      ['F','B','RL','RR'].forEach(id=>{
        hold(document.getElementById(id),id);
      });
    }
  </script>
  </head>
  <body>
    <h2>Carrito Tanque (1 L298N)</h2>
    <div>
      <button id='F'>&#8593;</button>
    </div>
    <div>
      <button class="rotate" id='RL'>&#8634; R-L</button>
      <button class="rotate" id='RR'>&#8635; R-R</button>
    </div>
    <div>
      <button id='B'>&#8595;</button>
    </div>
    <br>
    <button id="stop" onclick="sendCmd('S')">STOP</button>
    <p class="note">Sin strafe ni diagonales: con 1 solo L298N (2 canales) las
    ruedas van agrupadas por lado, no se pueden controlar las 4 por separado.</p>
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
  else if (move == "RL") rotateLeft();
  else if (move == "RR") rotateRight();
  else stopCar();

  server.send(200, "text/plain", "OK");
}

// ===== Motor Control Logic =====
// Solo 2 canales disponibles: no hay forma de mover el motor delantero distinto
// del trasero en un mismo lado, asi que strafe y diagonales no existen en esta
// variante (ver README.md de esta carpeta).
void stopCar(){
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
}

void forward(){
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  // lado izquierdo adelante
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  // lado derecho adelante
}

void backward(){
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH);  // lado izquierdo atras
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH);  // lado derecho atras
}

void rotateRight(){ // pivote sobre el sitio, sentido horario
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  // lado izquierdo adelante
  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); // lado derecho atras
}

void rotateLeft(){ // pivote sobre el sitio, sentido antihorario
  digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); // lado izquierdo atras
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  // lado derecho adelante
}
