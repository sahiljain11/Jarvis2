import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQml 2.0
import QtGraphicalEffects 1.12
import QtQuick.Layouts 1.12
import "../components"

JarvisWidget{

    Item {
        width: parent.width
        height: parent.height

        Image{
            id: back
            source: "frame3.png"
            anchors.fill: parent
            smooth: true
            opacity: 1
            focus: true


            Rectangle{
                z: -2
                anchors.fill: parent
                id: rect_col
                color: "#00FFF5"
                opacity: 0.4
            }
        }


        FontLoader {
            id: webFont
            source: "Orbitron-Medium.ttf"
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 20
            anchors.top: parent.top

            TextField {
                id: city_tf
                Layout.topMargin: 40
                placeholderText: qsTr("City")
                Layout.alignment: Qt.AlignHCenter
                font.pointSize:14
                font.family: webFont.name
                selectByMouse: true
                color: "#603140"
            }

            Butt {
                width: rect.width
                height: rect.height

                Rectangle {
                    id: rect
                    implicitWidth: 100
                    implicitHeight: 40
                    color: "white"
                }

                Text {
                    text: "Search"
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.verticalCenter: parent.verticalCenter
                    font.bold: true
                    font.family: webFont.name
                    color: "#603140"
                }

                Layout.alignment: Qt.AlignHCenter

                onTouched: {
                    console.log("touched")
                    weather.update_by_city(city_tf.text)
                }
            }

            Label{
                Layout.alignment: Qt.AlignHCenter
                id: temperature_lbl
                color: "white"
                font.family: webFont.name
                font { family: 'Helvetica'; pixelSize: 20; capitalization: Font.SmallCaps }
            }

            Image {
                id: sky_td
                Layout.preferredWidth: 60
                Layout.preferredHeight: 60
                anchors.horizontalCenter: parent.horizontalCenter
            }

            Label{
                id: checking
                color: "white"
            }

            Item {
                Layout.fillHeight: true
            }
        }

        Connections {
            target: weather
            onDataChanged:{
                if(!weather.hasError()){

                    try{
                        var temperature = weather.data['main']['temp']
                        var main = weather.data['weather'][0]['main']
                        var humidity = weather.data['main']['humidity']
                        var wind_speed = weather.data['wind']['speed']
                        temperature_lbl.text = " Temperature : " + Math.round(((temperature - 273.15) * 1.8) + 32) + " degrees Fahrenheit \n"  + " Current Conditions : " + main + "\n Humidity : " + humidity + "\n Wind Speed : " + wind_speed + " m/s"
                        var sunrise = weather.data['sys']['sunrise']
                        var sunset = weather.data['sys']['sunset']
                        var now = weather.data['dt']
                        if (now >= sunset)
                            rect_col.color = "#0431d1"
                        else if (now <= sunrise)
                            rect_col.color = "#0431d1"
                        else
                            rect_col.color = "#2bf2fc"
                        var oof = weather.data['weather'][0]['icon']
                        sky_td.source = "icons/" + oof + ".png"
                    }

                    catch (error) {
                        temperature_lbl.text = "oof sorry bruh"
                    }
                }
                else{
                    temperature_lbl.text = "error"}

            }
        }
    }
}