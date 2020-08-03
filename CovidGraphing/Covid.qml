import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.12
import QtGraphicalEffects 1.12
import QtCharts 2.3
import QtPositioning 5.12
import QtLocation 5.12
import QtQml 2.12
import "../components"
import "."


ApplicationWindow{
visibility: "Maximized"
JarvisWidget{
    id: wid
    width: 700*1.67
    height: 700

    signal graphChanged()
    Rectangle{
        width:  parent.width
        height: parent.height
        color: "#1e1e1e"
        opacity: 1

        // Placeholder for coronavirus news/symptoms bar 
        Rectangle{
            id: info
            //Position coronavirus news bar at the top of widget
            anchors{
                top: parent.top
                topMargin: parent.height/40
                left: parent.left
                leftMargin: parent.width/40
                right: parent.right
                rightMargin: parent.width/40
            }
            height: parent.height/12
            color:"white"
        }

        // Shows the corona graphs
        ListView{
            id: root
            //Set visual parameters
            snapMode: ListView.SnapOneItem
            highlightRangeMode: ListView.StrictlyEnforceRange
            highlightMoveDuration: 250
            orientation: ListView.Horizontal
            boundsBehavior: Flickable.StopAtBounds
            clip: true

            anchors{
                right: parent.right
                rightMargin: parent.width/40
                left: parent.horizontalCenter
                leftMargin: parent.width/20
                top: info.bottom
                topMargin: parent.height/40
                bottom: chart_search.top
                bottomMargin: parent.height/30
            }

            model: corona_graphs

            delegate: Component {
                SplineGraph{
                    id: cases_per_country
                    title: "Cases per Country"
                    antialiasing: true
                    country: coun
                    type: graph_type
                    theme: ChartView.ChartThemeDark
                    width: root.width
                    height: root.height
                }
            }
        }

        ListModel{
            id: corona_graphs
        }

        // Map plugin for the corona map
        Plugin{
            id: mapPlugin
            name: "esri"

            PluginParameter{
                name: "mapbox.access_token"
                value: "pk.eyJ1IjoiY29ucmFkbGlzdGUiLCJhIjoiY2tjMDBmcHBiMWVtbDJ1cDNjd2t0cnoxZyJ9.ryfkaChR_QGeAUyVfC6QSw"
            }
        }


        // Placeholder for corona map
        Map {
            id: map

            // Positions the coronavirus map
            anchors{
                right: root.left
                rightMargin: parent.width/40
                left: parent.left
                leftMargin: parent.width/40
                top: info.bottom
                topMargin: parent.height/40
                bottom: map_butt.top
                bottomMargin: parent.height/30
            }

            plugin: mapPlugin
            activeMapType: supportedMapTypes[0]

            Component.onCompleted:{
                for(var i = 0; i < supportedMapTypes.length; i++)
                    console.log(supportedMapTypes[i].name)
            }
        }
    

        //Searchbar for graph button
        TextField{
            id: chart_search
            anchors{
                horizontalCenter: root.horizontalCenter
                bottom: parent.bottom
                bottomMargin: parent.height/20
            }
            
            width: parent.width/8
            height: parent.height/20

            Keys.onPressed:{
                if(event.key == Qt.Key_Return){
                    corona_graphs.clear()
                    corona_graphs.append({coun: chart_search.text, graph_type: "cases"})
                    corona_graphs.append({coun: chart_search.text, graph_type: "deathes"})
                }
            }
        }

        //Placeholder for map button
        Rectangle{
            id: map_butt
            
            anchors{
                horizontalCenter: map.horizontalCenter
                bottom: parent.bottom
                bottomMargin: parent.height/20
            }

            width: parent.width/8
            height: parent.height/20
        }
    }
}
}