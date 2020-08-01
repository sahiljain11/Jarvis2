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
    width: 700*1.67
    height: 700
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

        // Placeholder for corona graph
        ChartView {
            id: graph
            title: "Cases per Country"
            antialiasing: true
            theme: ChartView.ChartThemeDark
            
            //Animates the grid lines
            animationOptions: {ChartView.GridAxisAnimations}
            
            //Controls the legend

            // Position the graph on the left under the news bar
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

            // One country
            SplineSeries {
                id: graphData
                name: ""

                axisX: DateTimeAxis {
                            id: xAxis
                            format: "MMM yyyy"
                        }

                axisY: ValueAxis{
                            id: yAxis
                            min: 0
                            max: 1000000
                        }

                Component.onCompleted:{
                    axisY.applyNiceNumbers()
                }
            }

            /*Component.onCompleted:{
                graph.addNewSeries(corona.countryallconfirmed('Russia'))
                //graphData.append(1,2)
            }*/

            function addNewSeries(country){
                graphData.removePoints(0, graphData.count)
                console.log(graphData.count)
                var data = corona.countryallconfirmed(country)
                if (data == null){
                    return -1
                }
                var max = 0
                var i = 0
                var start = 0
                var end = 0
                var date = 0
                for(var key in data){
                    var value = data[key]

                    if (value > max){
                        max = value
                    }
                    
                    date = new Date(key)
                    
                    if (i == 0){
                        start = date
                    }

                    graphData.append(date, value)
                    i++
                }
                end = date
                yAxis.max = max + 10000
                xAxis.min = start
                xAxis.max = end
                graphData.index
                return 0
            }
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
                right: graph.left
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
                horizontalCenter: graph.horizontalCenter
                bottom: parent.bottom
                bottomMargin: parent.height/20
            }
            
            width: parent.width/8
            height: parent.height/20

            Keys.onPressed:{
                if(event.key == Qt.Key_Return){
                    if (graph.addNewSeries(chart_search.text) == 0){
                        graphData.name = chart_search.text
                    }
                    else{
                        graphData.name = ""
                    }
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