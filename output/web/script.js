// 散布図の描画
document.addEventListener('DOMContentLoaded', function() {
    const scatterData = [{"cluster": 0, "label": "テント芝居の演出と表現技法", "color": "#1f77b4", "x": [-0.22360679774997902], "y": [0.7928352695954454], "text": ["「新装版 内側の時間」は、野らぼうが目標としてきたテント芝居を遂に実現した作品。言葉の奔流でありながら無駄のない台詞構成で、リアリティから始まり徐々に抽象度を高める構成は問答無用に面白い。成田明加の玉ねぎ丸かじりの演技は生命力を感じる山場だった。背景装置と自由に動くちゅみの存在が舞台を適温に保ち、全ての演者が素晴らしいアンサンブルを見せていた。"]}, {"cluster": 1, "label": "作品のコピーと価値観に対する批判的視点", "color": "#ff7f0e", "x": [-0.22360679774997907], "y": [-0.14027642872225954], "text": ["「どうしようもない価値に真剣になる大人」というコピーに怒りを感じた。現代社会の無価値の循環や意識の硬直に対する批判。「価値」という言葉自体への考察が足りていないと感じる。演劇をするなら言葉に敏感になるべきだ。出演者は「褒められたい」「認められたい」という欲求が見え、「どうしようもない価値に真剣になる」キャラを獲得して価値を得ようとするのはズルいと思う。"]}, {"cluster": 2, "label": "抽象的・詩的な表現による感想", "color": "#2ca02c", "x": [0.8944271909999157], "y": [-2.307253111968249e-16], "text": ["日常生活と芝居との回廊。ネバネバするがサラッとしている。繰り返しの襞は毎回同じで毎回違う。思考の粒子は自由になり、感性も感情も様々な場所に存在する。喜怒哀楽の楽しみ方は観客自身から溢れ出る。この芝居を観た人々は自立していく。"]}, {"cluster": 3, "label": "個人的な人生の省察と共感", "color": "#d62728", "x": [-0.22360679774997902], "y": [-0.5897355253474185], "text": ["芝居は家から出るところから始まった。浜松から松本までの道中が野らぼうだった。親とのしがらみ、結婚、日常の些細なことが走馬灯のように描かれる。芝居を観ているのか自分の人生を観ているのか。芝居後のため息は心地よく、人生は悔いだらけでも満更でもないと思わせる。こんな凄いものを日本全国民に見せたいと感じた。"]}, {"cluster": 4, "label": "生命と非生命の境界に関する哲学的考察", "color": "#9467bd", "x": [-0.223606797749979], "y": [-0.06282331552576832], "text": ["この芝居は目を閉じることができる劇だった。生命と非生命について考えさせられる内容で、時代の問いかけを感じた。生命の定義とは何か、情報のエネルギーを受けて動くようになったものが動物であり、遠く離れていても動きに影響し合う存在なのではないか。"]}];
    
    const traces = scatterData.map(cluster => {
        return {
            x: cluster.x,
            y: cluster.y,
            mode: 'markers',
            type: 'scatter',
            name: cluster.label,
            text: cluster.text,
            hoverinfo: 'text',
            marker: {
                size: 10,
                color: cluster.color,
                opacity: 0.7
            }
        };
    });
    
    const layout = {
        title: '意見の分布',
        hovermode: 'closest',
        xaxis: {
            title: 'X(意見の主要な特徴の違い - 肯定的/批判的、具体的/抽象的など)',
            zeroline: false,
            showgrid: false
        },
        yaxis: {
            title: 'Y(意見の二次的な特徴の違い - 個人的/一般的、感情的/分析的など)',
            zeroline: false,
            showgrid: false
        },
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 80,
            pad: 4
        },
        legend: {
            x: 0,
            y: 1,
            traceorder: 'normal',
            font: {
                family: 'sans-serif',
                size: 12,
                color: '#000'
            },
            bgcolor: '#E2E2E2',
            bordercolor: '#FFFFFF',
            borderwidth: 2
        }
    };
    
    Plotly.newPlot('scatter-plot', traces, layout, {responsive: true});
});

// クラスターカードのインタラクション
document.addEventListener('DOMContentLoaded', function() {
    const clusterCards = document.querySelectorAll('.cluster-card');
    
    clusterCards.forEach(card => {
        card.addEventListener('click', function() {
            // カードがクリックされたときの処理（必要に応じて）
        });
    });
});
